#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// Synthetic VQE -> Quantum-Controller Workload Model
//
// Purpose:
//   Model event frequency and memory traffic caused by a VQE workload,
//   without pretending that GPU-simulator traffic equals real QPU traffic.
//
// Current abstraction:
//   VQE objective
//     -> 1 base energy evaluation
//     -> 2 * N_params shifted evaluations (central difference)
//     -> each observe()
//        -> G measurement groups
//        -> S shots per group
//        -> command + parameter + readout traffic
//
// IMPORTANT:
//   - This is NOT a physical QPU simulator.
//   - This is NOT a QICK instruction-accurate model.
//   - Payload sizes and grouping/shots are configurable assumptions.
//   - In gem5 SE mode, all buffers are ordinary process memory.
//     They emulate traffic patterns, not distinct physical memories/devices.
// ============================================================

namespace {

volatile std::uint64_t global_sink = 0;

enum class ParamPolicy {
    FullVector,
    DeltaOnly
};

enum class ReadoutPolicy {
    ShotLevel,
    GroupAggregated
};

struct Config {
    // Algorithm-level parameters
    std::size_t n_objectives = 26;
    std::size_t n_params = 36;

    // Measurement-level parameters
    std::size_t groups = 5;      // Example assumption, not a universal truth
    std::size_t shots = 100;     // Keep default modest for gem5 runtime

    // Synthetic transfer sizes
    std::size_t full_param_bytes = 36 * sizeof(double); // 288 B for 36 doubles
    std::size_t delta_param_bytes = 16;                 // synthetic assumption
    std::size_t command_bytes = 32;                     // synthetic assumption
    std::size_t readout_bytes_per_shot = 8;             // synthetic assumption
    std::size_t aggregated_readout_bytes = 64;          // synthetic assumption

    ParamPolicy param_policy = ParamPolicy::FullVector;
    ReadoutPolicy readout_policy = ReadoutPolicy::GroupAggregated;

    // Whether to perform actual memory accesses.
    // false = only print analytical event/byte counts.
    bool execute_memory_traffic = true;
};

struct Stats {
    std::uint64_t objective_calls = 0;
    std::uint64_t base_observes = 0;
    std::uint64_t shifted_observes = 0;
    std::uint64_t total_observes = 0;

    std::uint64_t group_submissions = 0;
    std::uint64_t shot_executions = 0;

    std::uint64_t param_transfer_events = 0;
    std::uint64_t command_transfer_events = 0;
    std::uint64_t readout_transfer_events = 0;

    std::uint64_t param_bytes = 0;
    std::uint64_t command_bytes = 0;
    std::uint64_t readout_bytes = 0;
};

// A simple reusable buffer pair representing host/controller-visible traffic.
struct TrafficBuffers {
    std::vector<std::uint8_t> host_tx;
    std::vector<std::uint8_t> controller_rx;

    std::vector<std::uint8_t> controller_tx;
    std::vector<std::uint8_t> host_rx;

    // Synthetic controller-local accumulation/readout state.
    // In gem5 SE this is still ordinary process memory.
    std::vector<std::uint8_t> controller_local;

    explicit TrafficBuffers(const Config &cfg) {
        const std::size_t tx_max = std::max({
            cfg.full_param_bytes,
            cfg.delta_param_bytes,
            cfg.command_bytes,
            std::size_t{1}
        });

        const std::size_t rx_max = std::max({
            cfg.readout_bytes_per_shot,
            cfg.aggregated_readout_bytes,
            std::size_t{1}
        });

        host_tx.resize(tx_max, 0x5A);
        controller_rx.resize(tx_max, 0);

        controller_tx.resize(rx_max, 0xA5);
        host_rx.resize(rx_max, 0);

        controller_local.resize(4096, 0);
    }
};

std::size_t parse_size(const std::string &s) {
    std::size_t pos = 0;
    auto value = std::stoull(s, &pos);
    if (pos != s.size())
        throw std::runtime_error("Invalid integer: " + s);
    return static_cast<std::size_t>(value);
}

bool parse_bool(const std::string &s) {
    if (s == "1" || s == "true" || s == "on")
        return true;
    if (s == "0" || s == "false" || s == "off")
        return false;
    throw std::runtime_error("Invalid boolean: " + s);
}

void print_usage(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Algorithm-level:\n"
        << "  --objectives=N             default 26\n"
        << "  --params=N                 default 36\n\n"
        << "Measurement-level:\n"
        << "  --groups=N                 default 5\n"
        << "  --shots=N                  default 100\n\n"
        << "Traffic assumptions:\n"
        << "  --param-policy=full|delta  default full\n"
        << "  --full-param-bytes=N       default 288\n"
        << "  --delta-param-bytes=N      default 16\n"
        << "  --command-bytes=N          default 32\n"
        << "  --readout-policy=shot|aggregate\n"
        << "                              default aggregate\n"
        << "  --readout-shot-bytes=N     default 8\n"
        << "  --readout-agg-bytes=N      default 64\n\n"
        << "Execution:\n"
        << "  --execute=0|1              default 1\n"
        << "  --help\n";
}

Config parse_args(int argc, char **argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        auto split = arg.find('=');
        if (split == std::string::npos)
            throw std::runtime_error("Expected --key=value, got: " + arg);

        const std::string key = arg.substr(0, split);
        const std::string value = arg.substr(split + 1);

        if (key == "--objectives")
            cfg.n_objectives = parse_size(value);
        else if (key == "--params") {
            cfg.n_params = parse_size(value);
            // If user changes only --params, keep full-vector size consistent
            // unless they explicitly override --full-param-bytes later.
            cfg.full_param_bytes = cfg.n_params * sizeof(double);
        } else if (key == "--groups")
            cfg.groups = parse_size(value);
        else if (key == "--shots")
            cfg.shots = parse_size(value);
        else if (key == "--full-param-bytes")
            cfg.full_param_bytes = parse_size(value);
        else if (key == "--delta-param-bytes")
            cfg.delta_param_bytes = parse_size(value);
        else if (key == "--command-bytes")
            cfg.command_bytes = parse_size(value);
        else if (key == "--readout-shot-bytes")
            cfg.readout_bytes_per_shot = parse_size(value);
        else if (key == "--readout-agg-bytes")
            cfg.aggregated_readout_bytes = parse_size(value);
        else if (key == "--execute")
            cfg.execute_memory_traffic = parse_bool(value);
        else if (key == "--param-policy") {
            if (value == "full")
                cfg.param_policy = ParamPolicy::FullVector;
            else if (value == "delta")
                cfg.param_policy = ParamPolicy::DeltaOnly;
            else
                throw std::runtime_error("param-policy must be full or delta");
        } else if (key == "--readout-policy") {
            if (value == "shot")
                cfg.readout_policy = ReadoutPolicy::ShotLevel;
            else if (value == "aggregate")
                cfg.readout_policy = ReadoutPolicy::GroupAggregated;
            else
                throw std::runtime_error(
                    "readout-policy must be shot or aggregate");
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }

    if (cfg.groups == 0)
        throw std::runtime_error("groups must be > 0");
    if (cfg.n_params == 0)
        throw std::runtime_error("params must be > 0");
    if (cfg.n_objectives == 0)
        throw std::runtime_error("objectives must be > 0");

    return cfg;
}

// Copy bytes and consume one destination byte so the traffic cannot
// trivially disappear under optimization.
void transfer_bytes(std::vector<std::uint8_t> &dst,
                    const std::vector<std::uint8_t> &src,
                    std::size_t bytes,
                    std::uint64_t sequence) {
    if (bytes == 0)
        return;

    if (bytes > dst.size() || bytes > src.size())
        throw std::runtime_error("Transfer exceeds allocated buffer");

    std::memcpy(dst.data(), src.data(), bytes);

    const std::size_t idx =
        static_cast<std::size_t>((sequence * 1315423911ULL) % bytes);
    global_sink += dst[idx];
}

// Synthetic controller-local work generated per shot.
// This intentionally does NOT model physical QPU computation.
// It only creates a small state/readout-buffer access.
void controller_local_shot_touch(std::vector<std::uint8_t> &buf,
                                 std::uint64_t shot_id) {
    const std::size_t idx =
        static_cast<std::size_t>((shot_id * 64ULL) % buf.size());

    buf[idx] = static_cast<std::uint8_t>(
        buf[idx] + static_cast<std::uint8_t>(shot_id & 0xFF));

    global_sink += buf[idx];
}

class Workload {
public:
    explicit Workload(Config cfg)
        : cfg_(cfg), buffers_(cfg) {
        params_.resize(cfg_.n_params, 0.0);

        // Deterministic initial parameters.
        for (std::size_t i = 0; i < params_.size(); ++i)
            params_[i] = static_cast<double>(i) * 0.01;
    }

    Stats run() {
        Stats s;
        for (std::size_t obj = 0; obj < cfg_.n_objectives; ++obj) {
            ++s.objective_calls;

            // ------------------------------------------------
            // Classical optimizer-side parameter update.
            // This is intentionally simple: it only changes
            // values so the parameter buffer is not constant.
            // ------------------------------------------------
            optimizer_update(obj);

            // ------------------------------------------------
            // 1) Base energy evaluation: observe(x)
            // ------------------------------------------------
            ++s.base_observes;
            run_observe(s, /*is_base=*/true, /*param_index=*/0,
                        /*shift_sign=*/0);

            // ------------------------------------------------
            // 2) Central-difference gradient:
            //    for each parameter i:
            //      observe(x + h e_i)
            //      observe(x - h e_i)
            // ------------------------------------------------
            constexpr double step = 1e-4;

            for (std::size_t i = 0; i < cfg_.n_params; ++i) {
                const double original = params_[i];

                params_[i] = original + step;
                ++s.shifted_observes;
                run_observe(s, /*is_base=*/false, i, +1);

                params_[i] = original - step;
                ++s.shifted_observes;
                run_observe(s, /*is_base=*/false, i, -1);

                params_[i] = original;
            }
        }

        return s;
    }

private:
    Config cfg_;
    TrafficBuffers buffers_;
    std::vector<double> params_;

    std::uint64_t transfer_seq_ = 0;
    std::uint64_t shot_seq_ = 0;

    void optimizer_update(std::size_t objective) {
        // Lightweight deterministic CPU-side update.
        // This is NOT meant to reproduce L-BFGS internals.
        const double scale = 1e-6 * static_cast<double>(objective + 1);

        for (std::size_t i = 0; i < params_.size(); ++i)
            params_[i] += scale * static_cast<double>(i + 1);

        // Consume one value to keep this path observable.
        global_sink += static_cast<std::uint64_t>(
            std::abs(params_[objective % params_.size()]) * 1e9);
    }

    std::size_t param_payload_bytes(bool is_base) const {
        if (cfg_.param_policy == ParamPolicy::FullVector)
            return cfg_.full_param_bytes;

        // Delta-only model:
        // - base observe sends full current parameter state
        // - shifted observe sends only one changed parameter/update record
        return is_base ? cfg_.full_param_bytes : cfg_.delta_param_bytes;
    }

    void emit_param_transfer(Stats &s, bool is_base) {
        const std::size_t bytes = param_payload_bytes(is_base);

        ++s.param_transfer_events;
        s.param_bytes += bytes;

        if (cfg_.execute_memory_traffic)
            transfer_bytes(buffers_.controller_rx, buffers_.host_tx,
                           bytes, transfer_seq_++);
    }

    void emit_command_transfer(Stats &s) {
        ++s.command_transfer_events;
        s.command_bytes += cfg_.command_bytes;

        if (cfg_.execute_memory_traffic)
            transfer_bytes(buffers_.controller_rx, buffers_.host_tx,
                           cfg_.command_bytes, transfer_seq_++);
    }

    void emit_shot_readout_transfer(Stats &s) {
        ++s.readout_transfer_events;
        s.readout_bytes += cfg_.readout_bytes_per_shot;

        if (cfg_.execute_memory_traffic)
            transfer_bytes(buffers_.host_rx, buffers_.controller_tx,
                           cfg_.readout_bytes_per_shot, transfer_seq_++);
    }

    void emit_aggregated_readout_transfer(Stats &s) {
        ++s.readout_transfer_events;
        s.readout_bytes += cfg_.aggregated_readout_bytes;

        if (cfg_.execute_memory_traffic)
            transfer_bytes(buffers_.host_rx, buffers_.controller_tx,
                           cfg_.aggregated_readout_bytes, transfer_seq_++);
    }

    void run_observe(Stats &s,
                     bool is_base,
                     std::size_t param_index,
                     int shift_sign) {
        (void)param_index;
        (void)shift_sign;

        ++s.total_observes;

        // Model policy:
        // one parameter/update payload per observe.
        emit_param_transfer(s, is_base);

        // One observe expands into G measurement groups.
        for (std::size_t g = 0; g < cfg_.groups; ++g) {
            ++s.group_submissions;

            // One synthetic controller command per measurement group.
            emit_command_transfer(s);

            // The controller/QPU executes S shots for this group.
            for (std::size_t shot = 0; shot < cfg_.shots; ++shot) {
                ++s.shot_executions;

                if (cfg_.execute_memory_traffic)
                    controller_local_shot_touch(
                        buffers_.controller_local, shot_seq_++);

                if (cfg_.readout_policy == ReadoutPolicy::ShotLevel)
                    emit_shot_readout_transfer(s);
            }

            // Alternative policy: controller aggregates S shots locally,
            // then returns one compact result per measurement group.
            if (cfg_.readout_policy == ReadoutPolicy::GroupAggregated)
                emit_aggregated_readout_transfer(s);
        }
    }
};

const char *to_string(ParamPolicy p) {
    return p == ParamPolicy::FullVector ? "full-vector" : "delta-only";
}

const char *to_string(ReadoutPolicy p) {
    return p == ReadoutPolicy::ShotLevel
               ? "shot-level"
               : "group-aggregated";
}

double mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_config(const Config &cfg) {
    const std::size_t observes_per_objective =
        1 + 2 * cfg.n_params;

    std::cout << "=== Configuration ===\n";
    std::cout << "Objectives              : " << cfg.n_objectives << "\n";
    std::cout << "Parameters              : " << cfg.n_params << "\n";
    std::cout << "Observe/objective       : "
              << observes_per_objective << "\n";
    std::cout << "Measurement groups G    : " << cfg.groups << "\n";
    std::cout << "Shots/group S           : " << cfg.shots << "\n";
    std::cout << "Parameter policy        : "
              << to_string(cfg.param_policy) << "\n";
    std::cout << "Readout policy          : "
              << to_string(cfg.readout_policy) << "\n";
    std::cout << "Execute memory traffic  : "
              << (cfg.execute_memory_traffic ? "yes" : "no") << "\n\n";

    std::cout << "Synthetic payload assumptions:\n";
    std::cout << "  full param payload    : "
              << cfg.full_param_bytes << " B\n";
    std::cout << "  delta param payload   : "
              << cfg.delta_param_bytes << " B\n";
    std::cout << "  command payload       : "
              << cfg.command_bytes << " B\n";
    std::cout << "  readout / shot        : "
              << cfg.readout_bytes_per_shot << " B\n";
    std::cout << "  aggregate / group     : "
              << cfg.aggregated_readout_bytes << " B\n\n";
}

void print_stats(const Stats &s) {
    const std::uint64_t total_bytes =
        s.param_bytes + s.command_bytes + s.readout_bytes;

    std::cout << "=== Workload Results ===\n";
    std::cout << "Objective calls          : "
              << s.objective_calls << "\n";
    std::cout << "Base observe()           : "
              << s.base_observes << "\n";
    std::cout << "Shifted observe()        : "
              << s.shifted_observes << "\n";
    std::cout << "Total observe()          : "
              << s.total_observes << "\n";
    std::cout << "Group submissions        : "
              << s.group_submissions << "\n";
    std::cout << "Shot executions          : "
              << s.shot_executions << "\n\n";

    std::cout << "Transfer events:\n";
    std::cout << "  parameter              : "
              << s.param_transfer_events << "\n";
    std::cout << "  command                : "
              << s.command_transfer_events << "\n";
    std::cout << "  readout                : "
              << s.readout_transfer_events << "\n\n";

    std::cout << "Synthetic traffic volume:\n";
    std::cout << "  parameter bytes        : "
              << s.param_bytes << " B ("
              << std::fixed << std::setprecision(3)
              << mib(s.param_bytes) << " MiB)\n";
    std::cout << "  command bytes          : "
              << s.command_bytes << " B ("
              << mib(s.command_bytes) << " MiB)\n";
    std::cout << "  readout bytes          : "
              << s.readout_bytes << " B ("
              << mib(s.readout_bytes) << " MiB)\n";
    std::cout << "  total bytes            : "
              << total_bytes << " B ("
              << mib(total_bytes) << " MiB)\n\n";

    std::cout << "Checksum sink            : "
              << global_sink << "\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        Config cfg = parse_args(argc, argv);

        print_config(cfg);

        Workload workload(cfg);
        Stats stats = workload.run();

        print_stats(stats);

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
