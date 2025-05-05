#include "contrib/cxxopts/cxxopts.hpp"
#include <CL/sycl.hpp>
#include <cxxopts/cxxopts.hpp>
#include <fmt/format.h>
#include <fmt/core.h>
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

// Print sycl::info::device_type into string
template <>
struct fmt::formatter<sycl::info::device_type> {
    constexpr auto
    parse(fmt::format_parse_context & ctx) -> fmt::format_parse_context::iterator
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(const sycl::info::device_type & obj, FormatContext & ctx) -> decltype(ctx.out())
    {
        if (obj == sycl::info::device_type::cpu)
            return fmt::format_to(ctx.out(), "CPU");
        else if (obj == sycl::info::device_type::gpu)
            return fmt::format_to(ctx.out(), "GPU");
        else if (obj == sycl::info::device_type::accelerator)
            return fmt::format_to(ctx.out(), "accelerator");
        else if (obj == sycl::info::device_type::custom)
            return fmt::format_to(ctx.out(), "custom");
        else
            return fmt::format_to(ctx.out(), "unknown");
    }
};

void
list_sycl_devices_brief()
{
    auto platforms = sycl::platform::get_platforms();
    if (platforms.size() > 0) {
        for (auto & platform : platforms) {
            auto devices = platform.get_devices();
            for (auto & device : devices) {
                fmt::print("{} {}: {} [{}]\n",
                           generate_device_hash(device),
                           device.get_info<sycl::info::device::device_type>(),
                           device.get_info<sycl::info::device::name>(),
                           device.get_info<sycl::info::device::vendor>());
            }
        }
    }
    else {
        fmt::print("No SYCL platform found");
    }
}

int
main(int argc, char * argv[])
{
    auto opts = build_opts();
    try {
        auto res = opts.parse(argc, argv);
        if (res.count("help"))
            print_help(opts.help());
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
