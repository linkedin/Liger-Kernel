#pragma once

namespace liger {

template <int Compute>
struct MoeCommGeometry;

template <>
struct MoeCommGeometry<90> {
	static constexpr int TileM = 128;
	static constexpr int NC = 4;
	static constexpr int RingStageScale = 1;
	static constexpr int RingStageExtra = 0;
};

template <>
struct MoeCommGeometry<100> {
	static constexpr int TileM = 128;
	static constexpr int NC = 4;
	static constexpr int RingStageScale = 1;
	static constexpr int RingStageExtra = 0;
};

inline constexpr int kMaxMoeCommTileM =
	MoeCommGeometry<100>::TileM > MoeCommGeometry<90>::TileM
		? MoeCommGeometry<100>::TileM
		: MoeCommGeometry<90>::TileM;
inline constexpr int kMaxMoeTunedCommStages = 8;
inline constexpr int kMaxMoeCommStagesSm90 =
	kMaxMoeTunedCommStages * MoeCommGeometry<90>::RingStageScale
		+ MoeCommGeometry<90>::RingStageExtra;
inline constexpr int kMaxMoeCommStagesSm100 =
	kMaxMoeTunedCommStages * MoeCommGeometry<100>::RingStageScale
		+ MoeCommGeometry<100>::RingStageExtra;
inline constexpr int kMaxMoeCommStages =
	kMaxMoeCommStagesSm100 > kMaxMoeCommStagesSm90
		? kMaxMoeCommStagesSm100
		: kMaxMoeCommStagesSm90;

static_assert(
	MoeCommGeometry<90>::TileM % MoeCommGeometry<90>::NC == 0);
static_assert(
	MoeCommGeometry<100>::TileM % MoeCommGeometry<100>::NC == 0);

} // namespace liger
