#pragma once

#include <fmt/format.h>
#include <fmt/core.h>
#include <CL/sycl.hpp>

// Print sycl::info::device_type into string
template <>
struct fmt::formatter<cl::sycl::info::device_type> {
    constexpr auto
    parse(fmt::format_parse_context & ctx) -> fmt::format_parse_context::iterator
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(const cl::sycl::info::device_type & obj, FormatContext & ctx) -> decltype(ctx.out())
    {
        if (obj == cl::sycl::info::device_type::cpu)
            return fmt::format_to(ctx.out(), "CPU");
        else if (obj == cl::sycl::info::device_type::gpu)
            return fmt::format_to(ctx.out(), "GPU");
        else if (obj == cl::sycl::info::device_type::accelerator)
            return fmt::format_to(ctx.out(), "accelerator");
        else if (obj == cl::sycl::info::device_type::custom)
            return fmt::format_to(ctx.out(), "custom");
        else
            return fmt::format_to(ctx.out(), "unknown");
    }
};

template <>
struct fmt::formatter<cl::sycl::info::global_mem_cache_type> {
    constexpr auto
    parse(fmt::format_parse_context & ctx) -> fmt::format_parse_context::iterator
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(const cl::sycl::info::global_mem_cache_type & obj,
           FormatContext & ctx) -> decltype(ctx.out())
    {
        if (obj == cl::sycl::info::global_mem_cache_type::none)
            return fmt::format_to(ctx.out(), "none");
        else if (obj == cl::sycl::info::global_mem_cache_type::read_only)
            return fmt::format_to(ctx.out(), "read-only");
        else if (obj == cl::sycl::info::global_mem_cache_type::read_write)
            return fmt::format_to(ctx.out(), "read-write");
        else
            return fmt::format_to(ctx.out(), "unknown");
    }
};

template <>
struct fmt::formatter<cl::sycl::info::local_mem_type> {
    constexpr auto
    parse(fmt::format_parse_context & ctx) -> fmt::format_parse_context::iterator
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(const cl::sycl::info::local_mem_type & obj, FormatContext & ctx) -> decltype(ctx.out())
    {
        if (obj == cl::sycl::info::local_mem_type::none)
            return fmt::format_to(ctx.out(), "none");
        else if (obj == cl::sycl::info::local_mem_type::local)
            return fmt::format_to(ctx.out(), "local");
        else if (obj == cl::sycl::info::local_mem_type::global)
            return fmt::format_to(ctx.out(), "global");
        else
            return fmt::format_to(ctx.out(), "unknown");
    }
};

template <>
struct fmt::formatter<cl::sycl::info::execution_capability> {
    constexpr auto
    parse(fmt::format_parse_context & ctx) -> fmt::format_parse_context::iterator
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(const cl::sycl::info::execution_capability & obj,
           FormatContext & ctx) -> decltype(ctx.out())
    {
        if (obj == cl::sycl::info::execution_capability::exec_kernel)
            return fmt::format_to(ctx.out(), "kernel");
        else if (obj == cl::sycl::info::execution_capability::exec_native_kernel)
            return fmt::format_to(ctx.out(), "native-kernel");
        else
            return fmt::format_to(ctx.out(), "unknown");
    }
};

template <>
struct fmt::formatter<cl::sycl::aspect> {
    constexpr auto
    parse(fmt::format_parse_context & ctx) -> fmt::format_parse_context::iterator
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(const cl::sycl::aspect & obj, FormatContext & ctx) -> decltype(ctx.out())
    {
        if (obj == cl::sycl::aspect::cpu)
            return fmt::format_to(ctx.out(), "cpu");
        else if (obj == cl::sycl::aspect::gpu)
            return fmt::format_to(ctx.out(), "gpu");
        else if (obj == cl::sycl::aspect::accelerator)
            return fmt::format_to(ctx.out(), "accelerator");
        else if (obj == cl::sycl::aspect::custom)
            return fmt::format_to(ctx.out(), "custom");
        else if (obj == cl::sycl::aspect::emulated)
            return fmt::format_to(ctx.out(), "emulated");
        else if (obj == cl::sycl::aspect::host_debuggable)
            return fmt::format_to(ctx.out(), "host-debuggable");
        else if (obj == cl::sycl::aspect::fp16)
            return fmt::format_to(ctx.out(), "fp16");
        else if (obj == cl::sycl::aspect::fp64)
            return fmt::format_to(ctx.out(), "fp64");
        else if (obj == cl::sycl::aspect::atomic64)
            return fmt::format_to(ctx.out(), "atomic64");
        else if (obj == cl::sycl::aspect::image)
            return fmt::format_to(ctx.out(), "image");
        else if (obj == cl::sycl::aspect::online_compiler)
            return fmt::format_to(ctx.out(), "online-compiler");
        else if (obj == cl::sycl::aspect::online_linker)
            return fmt::format_to(ctx.out(), "online-linker");
        else if (obj == cl::sycl::aspect::queue_profiling)
            return fmt::format_to(ctx.out(), "queue-profiling");
        else if (obj == cl::sycl::aspect::usm_device_allocations)
            return fmt::format_to(ctx.out(), "usm-device-allocations");
        else if (obj == cl::sycl::aspect::usm_host_allocations)
            return fmt::format_to(ctx.out(), "usm-host-allocations");
        else if (obj == cl::sycl::aspect::usm_atomic_host_allocations)
            return fmt::format_to(ctx.out(), "usm-atomic-host-allocations");
        else if (obj == cl::sycl::aspect::usm_shared_allocations)
            return fmt::format_to(ctx.out(), "usm-shared-allocations");
        else if (obj == cl::sycl::aspect::usm_atomic_shared_allocations)
            return fmt::format_to(ctx.out(), "usm-atomic-shared-allocations");
        else if (obj == cl::sycl::aspect::usm_system_allocations)
            return fmt::format_to(ctx.out(), "usm-system-allocations");
        else
            return fmt::format_to(ctx.out(), "unknown");
    }
};

template <>
struct fmt::formatter<cl::sycl::info::fp_config> {
    constexpr auto
    parse(fmt::format_parse_context & ctx) -> fmt::format_parse_context::iterator
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto
    format(const cl::sycl::info::fp_config & obj, FormatContext & ctx) -> decltype(ctx.out())
    {
        if (obj == cl::sycl::info::fp_config::denorm)
            return fmt::format_to(ctx.out(), "deform");
        else if (obj == cl::sycl::info::fp_config::inf_nan)
            return fmt::format_to(ctx.out(), "inf-nan");
        else if (obj == cl::sycl::info::fp_config::round_to_nearest)
            return fmt::format_to(ctx.out(), "round-to-nearest");
        else if (obj == cl::sycl::info::fp_config::round_to_zero)
            return fmt::format_to(ctx.out(), "round-to-zero");
        else if (obj == cl::sycl::info::fp_config::round_to_inf)
            return fmt::format_to(ctx.out(), "round-to-inf");
        else if (obj == cl::sycl::info::fp_config::fma)
            return fmt::format_to(ctx.out(), "fma");
        else if (obj == cl::sycl::info::fp_config::correctly_rounded_divide_sqrt)
            return fmt::format_to(ctx.out(), "correctly-rounded-divide-sqrt");
        else if (obj == cl::sycl::info::fp_config::soft_float)
            return fmt::format_to(ctx.out(), "soft-float");
        else
            return fmt::format_to(ctx.out(), "unknown");
    }
};
