#include "formatters.h"
#include "contrib/cxxopts/cxxopts.hpp"
#include <CL/sycl.hpp>
#include <cxxopts/cxxopts.hpp>
#include <fmt/format.h>
#include <string>

using namespace cl;

constexpr uint16_t fnv_prime = 0x1003;
constexpr uint16_t fnv_offset = 0x811C;

uint16_t
fnv1a_hash(const std::string & str)
{
    uint16_t hash = fnv_offset;
    for (char c : str) {
        hash ^= c;
        hash *= fnv_prime;
    }
    return hash;
}

// --

cxxopts::Options
build_opts()
{
    cxxopts::Options cmdln_opts("lssycl");
    cmdln_opts.add_option("", "h", "help", "Show this help page", cxxopts::value<bool>(), "");
    cmdln_opts.add_option("", "v", "verbose", "Verbose output", cxxopts::value<bool>(), "");
    return cmdln_opts;
}

void
print_help(const std::string & help)
{
    fmt::print(stdout, "{}", help);
}

std::string
generate_device_hash(const sycl::device & device)
{
    auto platform = device.get_platform();
    return fmt::format("{:04x}:{:04x}",
                       fnv1a_hash(platform.get_info<sycl::info::platform::name>()),
                       fnv1a_hash(device.get_info<sycl::info::device::name>()));
}

std::string
format_bytes(uint64_t bytes)
{
    std::vector<const char *> units = { "B", "KB", "MB", "GB", "TB", "PB", "EB" };
    int unit_idx = 0;
    auto size = static_cast<double>(bytes);

    while (size >= 1024 && unit_idx < units.size() - 1) {
        size /= 1024;
        ++unit_idx;
    }
    if (unit_idx > 0)
        return fmt::format("{:.1f} {}", size, units[unit_idx]);
    else
        return fmt::format("{} B", bytes);
}

std::string
format_yes_no(bool state)
{
    if (state)
        return { "yes" };
    else
        return { "no" };
}

template <typename T>
std::string
format_vector(const std::vector<T> & values)
{
    std::string str;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0)
            str += ", ";
        str += fmt::format("{}", values[i]);
    }
    return str;
}

void
print_no_sycl_platforms()
{
    fmt::print("No SYCL platform found");
}

void
print_device_title(const sycl::device & device)
{
    fmt::print("{} {}: {} [{}]\n",
               generate_device_hash(device),
               device.get_info<sycl::info::device::device_type>(),
               device.get_info<sycl::info::device::name>(),
               device.get_info<sycl::info::device::vendor>());
}

void
print_device_details(const sycl::device & device)
{
    fmt::print("        Version: {}\n", device.get_info<sycl::info::device::version>());
    fmt::print("        Driver version: {}\n",
               device.get_info<sycl::info::device::driver_version>());
    fmt::print("        Max compute units: {}\n",
               device.get_info<sycl::info::device::max_compute_units>());
    fmt::print("        Max work group size: {}\n",
               device.get_info<sycl::info::device::max_work_group_size>());
    fmt::print("        Max clock frequency: {} MHz\n",
               device.get_info<sycl::info::device::max_clock_frequency>());
    fmt::print("        Address bits: {}\n", device.get_info<sycl::info::device::address_bits>());
    fmt::print("        Aspects: {}\n",
               format_vector(device.get_info<sycl::info::device::aspects>()));
    fmt::print("        Max size of memory allocation: {}\n",
               format_bytes(device.get_info<sycl::info::device::max_mem_alloc_size>()));
    fmt::print("        Max parameter size: {}\n",
               format_bytes(device.get_info<sycl::info::device::max_parameter_size>()));
    fmt::print("        Global memory cache type: {}\n",
               device.get_info<sycl::info::device::global_mem_cache_type>());
    fmt::print("        Global memory cache line size: {} B\n",
               device.get_info<sycl::info::device::global_mem_cache_line_size>());
    fmt::print("        Global memory cache size: {}\n",
               format_bytes(device.get_info<sycl::info::device::global_mem_cache_size>()));
    fmt::print("        Global memory size: {}\n",
               format_bytes(device.get_info<sycl::info::device::global_mem_size>()));
    fmt::print("        Local memory type: {}\n",
               device.get_info<sycl::info::device::local_mem_type>());
    fmt::print("        Local memory size: {}\n",
               format_bytes(device.get_info<sycl::info::device::local_mem_size>()));
    fmt::print("        Error correction support: {}\n",
               format_yes_no(device.get_info<sycl::info::device::error_correction_support>()));
    fmt::print("        Profiling timer resolution: {} ns\n",
               device.get_info<sycl::info::device::profiling_timer_resolution>());
    fmt::print("        Execution capabilities: {}\n",
               format_vector(device.get_info<sycl::info::device::execution_capabilities>()));
    if (device.has(sycl::aspect::fp16)) {
        fmt::print("        Half precision (fp16) capabilities: {}\n",
                   format_vector(device.get_info<sycl::info::device::half_fp_config>()));
    }
    fmt::print("        Single precision (fp32) capabilities: {}\n",
               format_vector(device.get_info<sycl::info::device::single_fp_config>()));
    if (device.has(sycl::aspect::fp64)) {
        fmt::print("        Double precision (fp64) capabilities: {}\n",
                   format_vector(device.get_info<sycl::info::device::double_fp_config>()));
    }

    if (device.has(sycl::aspect::image)) {
        fmt::print("        Image:\n");
        fmt::print("                Max read image args: {}\n",
                   device.get_info<sycl::info::device::max_read_image_args>());
        fmt::print("                Max write image args: {}\n",
                   device.get_info<sycl::info::device::max_write_image_args>());
        fmt::print("                Max dimensions of 2D images (W x H): {}x{}\n",
                   device.get_info<sycl::info::device::image2d_max_width>(),
                   device.get_info<sycl::info::device::image2d_max_height>());
        fmt::print("                Max dimensions of 3D images (W x H x D): {}x{}x{}\n",
                   device.get_info<sycl::info::device::image3d_max_width>(),
                   device.get_info<sycl::info::device::image3d_max_height>(),
                   device.get_info<sycl::info::device::image3d_max_depth>());
        fmt::print("                Max buffer size: {}\n",
                   format_bytes(device.get_info<sycl::info::device::image_max_buffer_size>()));
        fmt::print("                Max samplers: {}\n",
                   device.get_info<sycl::info::device::max_samplers>());
    }

    // TOOD: more partitioning info if partition_max_sub_devices > 0
    fmt::print("        Maximum number of sub-devices: {}\n",
               device.get_info<sycl::info::device::partition_max_sub_devices>());
}

void
list_sycl_devices_brief()
{
    auto platforms = sycl::platform::get_platforms();
    if (platforms.size() > 0) {
        for (auto & platform : platforms) {
            auto devices = platform.get_devices();
            for (auto & device : devices)
                print_device_title(device);
        }
    }
    else
        print_no_sycl_platforms();
}

void
list_sycl_devices_verbose()
{
    auto platforms = sycl::platform::get_platforms();
    if (platforms.size() > 0) {
        for (auto & platform : platforms) {
            auto devices = platform.get_devices();
            for (auto & device : devices) {
                print_device_title(device);
                print_device_details(device);
                fmt::print("\n");
            }
        }
    }
    else
        print_no_sycl_platforms();
}

int
main(int argc, char * argv[])
{
    auto opts = build_opts();
    try {
        auto res = opts.parse(argc, argv);
        if (res.count("help"))
            print_help(opts.help());
        else if (res.count("verbose"))
            list_sycl_devices_verbose();
        else
            list_sycl_devices_brief();
        return 0;
    }
    catch (const cxxopts::exceptions::exception & e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        fmt::print(stdout, "{}", opts.help());
        return 1;
    }
    catch (sycl::exception & e) {
        fmt::print("SYCL exception caught: {}", e.what());
        return 1;
    }
    catch (...) {
        fmt::print("Unknown exception caught\n");
        return 1;
    }
}
