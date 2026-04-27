#pragma once

#include <cstdint>

namespace autohyu_msgs
{
namespace traffic_light
{

static constexpr std::uint16_t TYPE_RED_YELLOW_GREEN = 0;
static constexpr std::uint16_t TYPE_RED_YELLOW_LEFT = 1;
static constexpr std::uint16_t TYPE_RED_YELLOW_LEFT_GREEN = 2;
static constexpr std::uint16_t TYPE_PED_RED_GREEN = 3;
static constexpr std::uint16_t TYPE_YELLOW_YELLOW_YELLOW = 100;

static constexpr std::uint16_t COLOR_RED = 1;
static constexpr std::uint16_t COLOR_YELLOW = 4;
static constexpr std::uint16_t COLOR_RED_YELLOW = 5;
static constexpr std::uint16_t COLOR_GREEN = 16;
static constexpr std::uint16_t COLOR_RED_GREEN = 17;
static constexpr std::uint16_t COLOR_YELLOW_GREEN = 20;
static constexpr std::uint16_t COLOR_LEFT = 32;
static constexpr std::uint16_t COLOR_RED_LEFT = 33;
static constexpr std::uint16_t COLOR_YELLOW_LEFT = 36;
static constexpr std::uint16_t COLOR_GREEN_LEFT = 48;

}  // namespace traffic_light
}  // namespace autohyu_msgs
