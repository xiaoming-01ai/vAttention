#pragma once 

#include <inttypes.h>
#include <functional>
#include <utility>

#include "vattn.h"

VATTN_NAMESPACE_BEGIN

template <typename T, typename Compare = std::less<T> >
class Heap
{
public:
    Heap();
    // own memory
    Heap(int k);
    Heap(const Heap &heap);
    Heap(Heap &&heap) noexcept;
    ~Heap();

public:
    Heap &operator=(Heap &&heap) noexcept;

public:
    inline void reset(int k = -1);
    
    inline const T &top();
    
    inline void push(const T &elem);
    inline void push(T &&elem);
    
    inline void pop_and_push(const T &elem);
    inline void pop_and_push(T &&elem);
    inline void pop();

    inline int  capacity() const;
    inline int  heap_size() const;
    inline int  size() const;
    inline void set_size(int size);
    inline bool empty() const;
    inline bool full() const;
    
    inline T *data();
    
private:
    inline void adjust_heap(uint32_t last_pos);
    inline void adjust_heap_for_pop(uint32_t last_pos);
    inline void init(int k);
    inline void swap(int i, int j);

    uint64_t align(uint64_t v, uint64_t alignment)
    {
        return ((v + alignment - 1) / alignment) * alignment;
    }

private:
    const static int MAX_CAPACITY = 100000;
    
private:
    Compare comp_;

    int capacity_;
    // top k
    int _k;
    // current number of elements
    int size_;
    // only for order pop 
    int order_idx_;

    T *elems_;
    T *base_elems_;

};

template<typename T, typename Compare>
const int Heap<T, Compare>::MAX_CAPACITY;

template<typename T, typename Compare>
Heap<T, Compare>::Heap()
    : comp_(Compare())
{
    capacity_ = 0;
    _k = 0;
    size_ = 0;
    order_idx_ = 0;
    elems_ = nullptr;
    base_elems_ = nullptr;
}

template<typename T, typename Compare>
Heap<T, Compare>::Heap(int k)
    :comp_(Compare())
{
    if (k > MAX_CAPACITY) {
        k = MAX_CAPACITY;
    }

    init(k);
}

template<typename T, typename Compare>
Heap<T, Compare>::Heap(const Heap &heap)
    : comp_(Compare())
{
    capacity_ = heap.capacity_;;
    _k = heap._k;
    size_ = heap.size_;;
    order_idx_ = heap.order_idx_;;
    elems_ = nullptr;
    base_elems_ = nullptr;

    if (capacity_ > 0) {
        size_t byteSize = sizeof(T) * capacity_;
        byteSize = align(byteSize, 64);
        elems_ = (T *)std::aligned_alloc(64, byteSize);
        memcpy(elems_, heap.elems_, size_ * sizeof(T));
        base_elems_ = elems_ - 1;
    }
}

template<typename T, typename Compare>
Heap<T, Compare>::Heap(Heap &&heap) noexcept
    : comp_(Compare())
{
    capacity_ = heap.capacity_;;
    _k = heap._k;
    size_ = heap.size_;;
    order_idx_ = heap.order_idx_;;
    elems_ = heap.elems_;
    base_elems_ = heap.base_elems_;

    // reset heap
    heap.capacity_ = 0;
    heap._k = 0;
    heap.size_ = 0;
    heap.order_idx_ = 0;
    heap.elems_ = nullptr;
    heap.base_elems_ = nullptr;
}

template<typename T, typename Compare>
Heap<T, Compare> &Heap<T, Compare>::operator=(Heap &&heap) noexcept
{
    if (this != &heap) {
        std::free(elems_);

        capacity_ = heap.capacity_;;
        _k = heap._k;
        size_ = heap.size_;;
        order_idx_ = heap.order_idx_;;
        elems_ = heap.elems_;
        base_elems_ = heap.base_elems_;

        // reset heap
        heap.capacity_ = 0;
        heap._k = 0;
        heap.size_ = 0;
        heap.order_idx_ = 0;
        heap.elems_ = nullptr;
        heap.base_elems_ = nullptr;
    }

    return *this;
}

template<typename T, typename Compare>
Heap<T, Compare>::~Heap()
{
    if constexpr (!std::is_trivially_destructible_v<T>) {
        for (size_t i = 0; i < size_; ++i) {
            elems_[i].~T();
        }
    }
    std::free(elems_);
    elems_ = nullptr;
    base_elems_ = nullptr;
}

template<typename T, typename Compare>
inline const T &Heap<T, Compare>::top()
{
    return elems_[0];
}

template<typename T, typename Compare>
void Heap<T, Compare>::push(const T &elem)
{
    if (_k == size_) {
        if (!comp_(elems_[0], elem)) {
            return pop_and_push(elem);
        }
        return;
    }
    // heap is not full
    uint32_t pos = size_ + 1;
    while (pos > 1) {
        uint32_t fatherPos = pos >> 1;
        if (!comp_(base_elems_[fatherPos], elem)) {
            break;
        }

        base_elems_[pos] = std::move(base_elems_[fatherPos]);

        pos = fatherPos;
    }

    base_elems_[pos] = elem;

    size_++;
    return;
}

template<typename T, typename Compare>
void Heap<T, Compare>::push(T &&elem)
{
    if (_k == size_) {
        if (!comp_(elems_[0], elem)) {
            return pop_and_push(std::move(elem));
        }
        return;
    }
    // heap is not full
    uint32_t pos = size_ + 1;
    while (pos > 1) {
        uint32_t fatherPos = pos >> 1;
        if (!comp_(base_elems_[fatherPos], elem)) {
            break;
        }

        base_elems_[pos] = std::move(base_elems_[fatherPos]);

        pos = fatherPos;
    }

    base_elems_[pos] = std::move(elem);

    size_++;
    return;
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::pop_and_push(const T &elem)
{
    elems_[0] = elem;
    return adjust_heap(size_);
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::pop_and_push(T &&elem)
{
    elems_[0] = std::move(elem);
    return adjust_heap(size_);
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::pop()
{
    // index begin with 1
    uint32_t pos = 1;
    uint32_t child = pos << 1;
    uint32_t last_pos = size_ - 1;
    while (child <= last_pos) {
        uint32_t right_child = child + 1;
        // should walk to right child
        if ((right_child <= last_pos && 
             comp_(base_elems_[child], base_elems_[right_child]))) {
            child = right_child;
        }
        base_elems_[pos] = std::move(base_elems_[child]);
        pos = child;
        child = pos << 1;
    }

    while (pos > 1) {
        uint32_t fatherPos = pos >> 1;
        if (!comp_(base_elems_[fatherPos], base_elems_[size_])) {
            break;
        }

        base_elems_[pos] = std::move(base_elems_[fatherPos]);

        pos = fatherPos;
    }
    base_elems_[pos] = std::move(base_elems_[size_]);
    if constexpr (!std::is_trivially_destructible_v<T>) {
        base_elems_[size_].~T();
    }
    --size_;
    return;
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::reset(int k)
{
    if constexpr (!std::is_trivially_destructible_v<T>) {        
        for (size_t i = 0; i < size_; ++i) {
            elems_[i].~T();
        }
    }        
    
    if (k == -1) {
        size_ = 0;
        order_idx_ = 0;
        return;
    }

    if (k > MAX_CAPACITY) {
        k = MAX_CAPACITY;
    }

    if (capacity_ >= k + 1) {
        _k = k;
        size_ = 0;
        order_idx_ = 0;
        return;
    }

    std::free(elems_);

    init(k);

    return;
}

template<typename T, typename Compare>
inline int Heap<T, Compare>::capacity() const
{
    return capacity_;
}

template<typename T, typename Compare>
inline int Heap<T, Compare>::heap_size() const
{
    return _k;
}

template<typename T, typename Compare>
inline int Heap<T, Compare>::size() const
{
    return size_;
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::set_size(int newSize)
{
    size_ = newSize;
}

template<typename T, typename Compare>
inline bool Heap<T, Compare>::empty() const
{
    return size_ == 0;
}

template<typename T, typename Compare>
inline bool Heap<T, Compare>::full() const
{
    return size_ == _k;
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::adjust_heap_for_pop(uint32_t last_pos)
{
    T target_elem = std::move(elems_[0]);

    // index begin with 1
    uint32_t pos = 1;
    uint32_t child = pos << 1;
    while (child <= last_pos) {
        uint32_t right_child = child + 1;
        // should walk to right child
        if ((right_child <= last_pos && 
             comp_(base_elems_[child], base_elems_[right_child]))) {
            child = right_child;
        }
        base_elems_[pos] = std::move(base_elems_[child]);
        pos = child;
        child = pos << 1;
    }

    while (pos > 1) {
        uint32_t fatherPos = pos >> 1;
        if (!comp_(base_elems_[fatherPos], target_elem)) {
            break;
        }

        base_elems_[pos] = std::move(base_elems_[fatherPos]);

        pos = fatherPos;
    }
    base_elems_[pos] = std::move(target_elem);
    
    return;
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::adjust_heap(uint32_t last_pos)
{
    T target_elem = std::move(elems_[0]);

    // index begin with 1
    uint32_t pos = 1;
    uint32_t child = pos << 1;
    while (child <= last_pos) {
        uint32_t right_child = child + 1;
        // should walk to right child
        if ((right_child <= last_pos && 
             comp_(base_elems_[child], base_elems_[right_child]))) {
            child = right_child;
        }

        // if find the right position 
        if (!comp_(target_elem, base_elems_[child])) {
            break;
        }

        // continue to scan
        base_elems_[pos] = std::move(base_elems_[child]);
        pos = child;
        child = pos << 1;
    }

    base_elems_[pos] = std::move(target_elem);
    
    return;
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::init(int k)
{
    capacity_ = k + 1;
    _k = k;
    size_ = 0;
    order_idx_ = 0;
    size_t byteSize = sizeof(T) * capacity_;
    byteSize = align(byteSize, 64);
    capacity_ = byteSize / sizeof(T);
    elems_ = (T *)std::aligned_alloc(64, byteSize);    
    base_elems_ = elems_ - 1;

    return;
}

template<typename T, typename Compare>
inline void Heap<T, Compare>::swap(int i, int j)
{
    T elem = std::move(elems_[i]);
    elems_[i] = std::move(elems_[j]);
    elems_[j] = std::move(elem);

    return;
}
    
template<typename T, typename Compare>
inline T *Heap<T, Compare>::data()
{
    return elems_;
}

VATTN_NAMESPACE_END