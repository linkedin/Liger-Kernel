#pragma once

#include <cstdint>
#include <string>

namespace liger {

class MoeContextScope {
 public:
	explicit MoeContextScope(std::int64_t team);
	~MoeContextScope();
	MoeContextScope(const MoeContextScope&) = delete;
	MoeContextScope& operator=(const MoeContextScope&) = delete;
 private:
	std::int64_t previous_;
};

void moe_configure_context(int max_tokens, int hidden_dim, int max_num_experts,
	int max_top_k, int num_hosts, int gpus_per_host, int max_inflight,
	std::int64_t team, std::int64_t slot);
void reset_moe_configuration(bool clear_capacity = true);
void release_moe_team(std::int64_t team);
std::string moe_buffer_name(const char* name);
void moe_check_forward_capacity();

}  // namespace liger
