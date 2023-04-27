#ifndef _JUSTLM_POOL_HPP
#define _JUSTLM_POOL_HPP
#include "justlm.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <memory>
#include <optional>


namespace LM {
class InferencePool {
    class Slot {
        std::unique_ptr<Inference> inference;
        size_t id;
        std::chrono::system_clock::time_point last_access;
        std::string weights_path;

    public:

        Slot() {
            reset();
        }

        void reset() {
            inference = nullptr;
            id = 0;
        }
        bool is_free() const {
            return inference == nullptr;
        }
        Inference& create_inference(size_t id, const std::string& weights_path, const Inference::Params& p) {
            this->id = id;
            this->weights_path = weights_path;
            inference.reset(Inference::construct(weights_path, p));
            return get_inference(true);
        }
        Inference& get_inference(bool update_last_access = false) {
            if (update_last_access) last_access = std::chrono::system_clock::now();
            return *inference.get();
        }

        auto get_id() const {
            return id;
        }
        auto get_last_access() const {
            return last_access;
        }
        std::string_view get_weights_path() const {
            return weights_path;
        }
    };
    std::vector<Slot> slots;

    std::string pool_name;
    bool store_on_destruct = false;

    std::string get_slot_filename_prefix() const {
        return "LMInferencePool_"+pool_name+'_';
    }
    std::string get_slot_filename(size_t id) const {
        return get_slot_filename_prefix()+std::to_string(id);
    }

    // Returns false on error
    bool store_slot(Slot& slot);
    // Returns nullptr on error
    Slot *load_slot(size_t id, Slot *suggested_slot = nullptr);

    void store_and_reset_slot(Slot& slot) {
        store_slot(slot); //TODO: Should handle errors somehow
        slot.reset();
    }

    // Doesn't fail
    Slot& get_free_slot();

    // Returns nullptr if not found
    Slot* find_slot_by_id(size_t id, bool deserialize = true);

public:
    // The pool_name must be unique amonst all applications in cwd
    InferencePool(size_t size, const std::string& pool_name, bool clean_up = true);
    ~InferencePool() {
        if (store_on_destruct) {
            store_all();
        }
    }

    Inference &create_inference(size_t id, const std::string& weights_path, const Inference::Params& p) {
        auto& slot = get_free_slot();
        return slot.create_inference(id, weights_path, p);
    }
    std::optional<std::reference_wrapper<Inference>> get_inference(size_t id);
    Inference &get_or_create_inference(size_t id, const std::string& weights_path, const Inference::Params& p);
    void delete_inference(size_t id);
    void store_all();
    std::vector<size_t> get_active_slot_ids() const;

    void set_store_on_destruct(bool v) {
        store_on_destruct = v;
    }
    bool is_stored_on_destruction() const {
        return store_on_destruct;
    }
};
}
#endif // _JUSTLM_POOL_HPP