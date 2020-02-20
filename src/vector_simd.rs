// Copyright 2013-2014 The CGMath Developers. For a full listing of the authors,
// refer to the Cargo.toml file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use structure::ElementWise;
use vector::*;

use std::ops::*;

use num::BaseNum;
use packed_simd::{Simd, SimdArray};

impl<S> From<Simd<[S; 4]>> for Vector4<S>
where
    S: Copy,
    [S; 4]: SimdArray + From<Simd<[S; 4]>>,
{
    #[inline]
    fn from(f: Simd<[S; 4]>) -> Self {
        let arr: [S; 4] = f.into();
        Vector4 {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            w: arr[3],
        }
    }
}

impl<S> Into<Simd<[S; 4]>> for Vector4<S>
where
    [S; 4]: SimdArray + Into<Simd<[S; 4]>>,
{
    #[inline]
    fn into(self) -> Simd<[S; 4]> {
        let arr: [S; 4] = [self.x, self.y, self.z, self.w];
        arr.into()
    }
}

impl<S> From<Simd<[S; 4]>> for Vector3<S>
where
    S: Copy,
    [S; 4]: SimdArray + From<Simd<[S; 4]>>,
{
    #[inline]
    fn from(f: Simd<[S; 4]>) -> Self {
        let arr: [S; 4] = f.into();
        Vector3 {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl<S> Into<Simd<[S; 4]>> for Vector3<S>
where
    S: BaseNum,
    [S; 4]: SimdArray + Into<Simd<[S; 4]>>,
{
    #[inline]
    fn into(self) -> Simd<[S; 4]> {
        let arr: [S; 4] = [self.x, self.y, self.z, S::zero()];
        arr.into()
    }
}

impl<S> From<Simd<[S; 2]>> for Vector2<S>
where
    S: Copy,
    [S; 2]: SimdArray + From<Simd<[S; 2]>>,
{
    #[inline]
    fn from(f: Simd<[S; 2]>) -> Self {
        let arr: [S; 2] = f.into();
        Vector2 {
            x: arr[0],
            y: arr[1],
        }
    }
}

impl<S> Into<Simd<[S; 2]>> for Vector2<S>
where
    S: Copy,
    [S; 2]: SimdArray + Into<Simd<[S; 2]>>,
{
    #[inline]
    fn into(self) -> Simd<[S; 2]> {
        let arr: [S; 2] = [self.x, self.y];
        arr.into()
    }
}

// Map a VectorN<S> to an array type that can be converted into a SIMD type.
// Useful for adding bounds on SimdArray trait from the packed_simd crate.
macro_rules! arr_t(
    (Vector4<$S:ty>) => { [$S; 4] };
    (Vector3<$S:ty>) => { [$S; 4] };
    (Vector2<$S:ty>) => { [$S; 2] };
);

// Map a VectorN<S> into a Simd<[N; S]> type, where N is the size of the vector.
macro_rules! simd_t(
    ($VectorN:ident<$S:ty>) => { Simd<arr_t!($VectorN<$S>)> };
);

macro_rules! impl_extra_simd_ops(
    ($VectorN:ident<$S:ty>) => {
        impl $VectorN<$S> {
            /// Compute and return the square root of each element.
            #[inline]
            pub fn sqrt_element_wide(self) -> Self {
                let s: simd_t!($VectorN<$S>) = self.into();
                s.sqrt().into()
            }

            /// Compute and return the reciprocal of the square root of each element.
            #[inline]
            pub fn rsqrt_element_wide(self) -> Self {
                let s: simd_t!($VectorN<$S>) = self.into();
                s.rsqrte().into()
            }

            /// Compute and return the reciprocal of each element.
            #[inline]
            pub fn recip_element_wide(self) -> Self {
                let s: simd_t!($VectorN<$S>) = self.into();
                s.recpre().into()
            }
        }
    }
);

impl_extra_simd_ops!(Vector4<f32>);
impl_extra_simd_ops!(Vector4<f64>);
impl_extra_simd_ops!(Vector3<f32>);
impl_extra_simd_ops!(Vector3<f64>);
impl_extra_simd_ops!(Vector2<f32>);
impl_extra_simd_ops!(Vector2<f64>);

macro_rules! impl_operator_simd2 {
    // *assign ops
    (impl<S: $Constraint:ident> $Op:ident for $VectorN:ident<S>: $BaseOp:ident {
        fn $op:ident($lhs:ident, $rhs:ident) { *self = $body:expr; }
    }) => {
        impl<S> $Op for $VectorN<S>
        where
            S: BaseNum,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $BaseOp<Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(&mut self, other: $VectorN<S>) {
                let $lhs: simd_t!($VectorN<S>) = (*self).into();
                let $rhs: simd_t!($VectorN<S>) = other.into();
                *self = $body;
            }
        }
    };

    // *assign ops scalar type
    (impl<S: $Constraint:ident> $Op:ident<S> for $VectorN:ident<S>: $BaseOp:ident {
        fn $op:ident($lhs:ident, $rhs:ident) { *self = $body:expr; }
    }) => {
        impl<S> $Op<S> for $VectorN<S>
        where
            S: BaseNum,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $BaseOp<S, Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(&mut self, $rhs: S) {
                let $lhs: simd_t!($VectorN<S>) = (*self).into();
                *self = $body;
            }
        }
    };

    // rhs is a scalar
    (impl<S: $Constraint:ident> $Op:ident<$Rhs:ident> for $VectorN:ident<S> {
        fn $op:ident($lhs:ident, $rhs:ident) -> $Output:ty { $body:expr }
    }) => {
        impl<S> $Op<$Rhs> for $VectorN<S>
        where
            S: $Constraint,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $Op<S, Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(self, $rhs: $Rhs) -> $Output {
                let $lhs: simd_t!($VectorN<S>) = self.into();
                $body
            }
        }

        impl<'a, S> $Op<$Rhs> for &'a $VectorN<S>
        where
            S: $Constraint,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $Op<S, Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(self, $rhs: $Rhs) -> $Output {
                let $lhs: simd_t!($VectorN<S>) = (*self).into();
                $body
            }
        }
    };

    // vector operations (element wise) (RHS is a compound type)
    (impl<S: $Constraint:ident> $Op:ident<$Rhs:ty> for $VectorN:ident<S> {
        fn $op:ident($lhs:ident, $rhs:ident) -> $Output:ty { $body:expr }
    }) => {
        impl<S> $Op<$Rhs> for $VectorN<S>
        where
            S: $Constraint,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $Op<Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
            $Rhs: Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(self, other: $Rhs) -> $Output {
                let $lhs: simd_t!($VectorN<S>) = self.into();
                let $rhs: simd_t!($VectorN<S>) = other.into();
                $body
            }
        }

        impl<'a, S> $Op<&'a $Rhs> for $VectorN<S>
        where
            S: $Constraint,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $Op<Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
            $Rhs: Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(self, other: &'a $Rhs) -> $Output {
                let $lhs: simd_t!($VectorN<S>) = self.into();
                let $rhs: simd_t!($VectorN<S>) = (*other).into();
                $body
            }
        }

        impl<'a, S> $Op<$Rhs> for &'a $VectorN<S>
        where
            S: $Constraint,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $Op<Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
            $Rhs: Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(self, other: $Rhs) -> $Output {
                let $lhs: simd_t!($VectorN<S>) = (*self).into();
                let $rhs: simd_t!($VectorN<S>) = other.into();
                $body
            }
        }

        impl<'a, 'b, S> $Op<&'a $Rhs> for &'b $VectorN<S>
        where
            S: $Constraint,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): $Op<Output = simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
            $Rhs: Into<simd_t!($VectorN<S>)>,
        {
            #[inline]
            fn $op(self, other: &'a $Rhs) -> $Output {
                let $lhs: simd_t!($VectorN<S>) = (*self).into();
                let $rhs: simd_t!($VectorN<S>) = (*other).into();
                $body
            }
        }
    };
}

macro_rules! impl_vector_simd {
    ($VectorN:ident) => {
        impl_operator_simd2!(
            impl<S: BaseNum> Add<$VectorN<S>> for $VectorN<S> {
                fn add(lhs, rhs) -> $VectorN<S> {
                    (lhs + rhs).into()
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> Sub<$VectorN<S>> for $VectorN<S> {
                fn sub(lhs, rhs) -> $VectorN<S> {
                    (lhs - rhs).into()
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> Mul<S> for $VectorN<S> {
                fn mul(lhs, rhs) -> $VectorN<S> {
                    (lhs * rhs).into()
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> Div<S> for $VectorN<S> {
                fn div(lhs, rhs) -> $VectorN<S> {
                    (lhs / rhs).into()
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> Rem<S> for $VectorN<S> {
                fn rem(lhs, rhs) -> $VectorN<S> {
                    (lhs % rhs).into()
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> AddAssign for $VectorN<S>: Add {
                fn add_assign(lhs, rhs) {
                    *self = (lhs + rhs).into();
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> SubAssign for $VectorN<S>: Sub {
                fn sub_assign(lhs, rhs) {
                    *self = (lhs - rhs).into();
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> MulAssign<S> for $VectorN<S>: Mul {
                fn mul_assign(lhs, rhs) {
                    *self = (lhs * rhs).into();
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> DivAssign<S> for $VectorN<S>: Div {
                fn div_assign(lhs, rhs) {
                    *self = (lhs / rhs).into();
                }
            }
        );

        impl_operator_simd2!(
            impl<S: BaseNum> RemAssign<S> for $VectorN<S>: Rem {
                fn rem_assign(lhs, rhs) {
                    *self = (lhs % rhs).into();
                }
            }
        );

        impl<S> Neg for $VectorN<S>
        where
            S: Neg<Output = S>,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>): Neg<Output=simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
        {
            fn neg(self) -> $VectorN<S> {
                let lhs: simd_t!($VectorN<S>) = self.into();
                (-lhs).into()
            }
        }

        impl<S> ElementWise for $VectorN<S>
        where
            S: BaseNum,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>):
                Mul<Output=simd_t!($VectorN<S>)> +
                Div<Output=simd_t!($VectorN<S>)> +
                Rem<Output=simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
        {
            fn add_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> {
                self + rhs
            }

            fn sub_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> {
                self - rhs
            }

            fn mul_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> {
                let lhs: simd_t!($VectorN<S>) = self.into();
                let rhs: simd_t!($VectorN<S>) = rhs.into();
                (lhs * rhs).into()
            }

            fn div_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> {
                let lhs: simd_t!($VectorN<S>) = self.into();
                let rhs: simd_t!($VectorN<S>) = rhs.into();
                (lhs / rhs).into()
            }

            fn rem_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> {
                let lhs: simd_t!($VectorN<S>) = self.into();
                let rhs: simd_t!($VectorN<S>) = rhs.into();
                (lhs % rhs).into()
            }

            fn add_assign_element_wise(&mut self, rhs: $VectorN<S>) {
                (*self) += rhs;
            }

            fn sub_assign_element_wise(&mut self, rhs: $VectorN<S>) {
                (*self) -= rhs;
            }

            fn mul_assign_element_wise(&mut self, rhs: $VectorN<S>) {
                let lhs: simd_t!($VectorN<S>) = (*self).into();
                let rhs: simd_t!($VectorN<S>) = rhs.into();
                *self = (lhs * rhs).into()
            }

            fn div_assign_element_wise(&mut self, rhs: $VectorN<S>) {
                let lhs: simd_t!($VectorN<S>) = (*self).into();
                let rhs: simd_t!($VectorN<S>) = rhs.into();
                *self = (lhs / rhs).into()
            }

            fn rem_assign_element_wise(&mut self, rhs: $VectorN<S>) {
                let lhs: simd_t!($VectorN<S>) = (*self).into();
                let rhs: simd_t!($VectorN<S>) = rhs.into();
                *self = (lhs % rhs).into()
            }
        }

        impl<S> ElementWise<S> for $VectorN<S>
        where
            S: BaseNum,
            arr_t!($VectorN<S>): SimdArray,
            simd_t!($VectorN<S>):
                Add<S, Output=simd_t!($VectorN<S>)> +
                Sub<S, Output=simd_t!($VectorN<S>)>,
            $VectorN<S>: From<simd_t!($VectorN<S>)> + Into<simd_t!($VectorN<S>)>,
        {
            fn add_element_wise(self, rhs: S) -> $VectorN<S> {
                let lhs: simd_t!($VectorN<S>) = self.into();
                (lhs + rhs).into()
            }

            fn sub_element_wise(self, rhs: S) -> $VectorN<S> {
                let lhs: simd_t!($VectorN<S>) = self.into();
                (lhs - rhs).into()
            }

            fn mul_element_wise(self, rhs: S) -> $VectorN<S> {
                self * rhs
            }

            fn div_element_wise(self, rhs: S) -> $VectorN<S> {
                self / rhs
            }

            fn rem_element_wise(self, rhs: S) -> $VectorN<S> {
                self % rhs
            }

            fn add_assign_element_wise(&mut self, rhs: S) {
                let lhs: simd_t!($VectorN<S>) = (*self).into();
                *self = (lhs + rhs).into();
            }

            fn sub_assign_element_wise(&mut self, rhs: S) {
                let lhs: simd_t!($VectorN<S>) = (*self).into();
                *self = (lhs + rhs).into();
            }

            fn mul_assign_element_wise(&mut self, rhs: S) {
                (*self) *= rhs;
            }

            fn div_assign_element_wise(&mut self, rhs: S) {
                (*self) /= rhs;
            }

            fn rem_assign_element_wise(&mut self, rhs: S) {
                (*self) %= rhs;
            }
        }
    };
}

impl_vector_simd!(Vector4);
impl_vector_simd!(Vector3);
impl_vector_simd!(Vector2);
