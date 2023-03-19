/*!
 * \file
 *
 * Defines number_type, and has some ifdeffed, but potentially useful code.
 *
 * This set of template meta programming incantations creates a number_type template,
 * which will set its value to 1 if its template argument is scalar or 0 if it is not
 * scalar. It would be possible to avoid the use of number_type in the classes which use
 * it (instead using is_scalar directly in those class templates), but it affords some
 * potential flexibility to do it this way.
 *
 * Previously, it set value to one of three integer values signifying whether it is a
 * resizable 'vector' type, such as std::vector or std::list (value=0), a fixed-size
 * 'vector' type, such as std::array (value=1) OR a scalar (value=2).
 */

#pragma once

#include <type_traits>

/*! \brief A class to distinguish between scalars and vectors
 *
 * From the typename T, set a #value attribute which says whether T is a scalar (like
 * float, double), or vector (basically, anything else).
 *
 * I did experiment with code (which is ifdeffed out in the file number_type.h) which
 * would determine whether \a T is a resizable list-like type (vector, list etc) or
 * whether it is a fixed-size list-like type, such as std::array. This was because I
 * erroneously thought I would have to have separate implementations for each. The
 * Ifdeffed code is left for future reference.
 *
 * \tparam T the type to distinguish
 */
template <typename T>
struct number_type {
    //! is_scalar test
    static constexpr bool const scalar = std::is_scalar<std::decay_t<T>>::value;
    //! Set value simply from the is_scalar test. 0 for vector, 1 for scalar
    static constexpr int const value = scalar ? 1 : 0;
#endif
};
