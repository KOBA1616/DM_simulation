#pragma once

#include <unordered_map>
#include <cstddef>

namespace dm::ai {

    struct MCTSNode;

    class TranspositionTable {
        std::unordered_map<size_t, MCTSNode*> hash_to_node;

    public:
        MCTSNode* lookup(size_t hash) {
            auto it = hash_to_node.find(hash);
            if (it != hash_to_node.end()) {
                return it->second;
            }
            return nullptr;
        }

        void store(size_t hash, MCTSNode* node) {
            hash_to_node[hash] = node;
        }

        void clear() {
            hash_to_node.clear();
        }
    };

}
