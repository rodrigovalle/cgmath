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
    fn into(self) -> Simd<[S; 4]> {
        let arr: [S; 4] = [self.x, self.y, self.z, self.w];
        arr.into()
    }
}

macro_rules! impl_extra_simd_ops(
    (Vector4<$S:ident>) => {
        impl Vector4<$S> {
            /// Compute and return the square root of each element.
            #[inline]
            pub fn sqrt_element_wide(self) -> Self {
                let s: Simd<[$S; 4]> = self.into();
                s.sqrt().into()
            }

            /// Compute and return the reciprocal of the square root of each element.
            #[inline]
            pub fn rsqrt_element_wide(self) -> Self {
                let s: Simd<[$S; 4]> = self.into();
                s.rsqrte().into()
            }

            /// Compute and return the reciprocal of each element.
            #[inline]
            pub fn recip_element_wide(self) -> Self {
                let s: Simd<[$S; 4]> = self.into();
                s.recpre().into()
            }
        }
    }
);

impl_extra_simd_ops!(Vector4<f32>);
impl_extra_simd_ops!(Vector4<f64>);

macro_rules! impl_operator_simd2 {
    (impl<S> $Op:ident<S> for Vector4<S> {
        fn $op:ident($lhs:ident, $rhs:ident) -> $Output:ty { $body:expr }
    }) => {
        impl<S> $Op<S> for Vector4<S>
        where
            S: BaseNum,
            [S; 4]: SimdArray,
            Simd<[S; 4]>: $Op<S, Output = Simd<[S; 4]>>,
            Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
        {
            fn $op(self, $rhs: S) -> $Output {
                let $lhs: Simd<[S; 4]> = self.into();
                $body
            }
        }

        impl<'a, S> $Op<S> for &'a Vector4<S>
        where
            S: BaseNum,
            [S; 4]: SimdArray,
            Simd<[S; 4]>: $Op<S, Output = Simd<[S; 4]>>,
            Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
        {
            fn $op(self, $rhs: S) -> $Output {
                let $lhs: Simd<[S; 4]> = (*self).into();
                $body
            }
        }
    };

    (impl<S> $Op:ident<Vector4<S>> for Vector4<S> {
        fn $op:ident($lhs:ident, $rhs:ident) -> $Output:ty { $body:expr }
    }) => {
        impl<S> $Op<Vector4<S>> for Vector4<S>
        where
            S: BaseNum,
            [S; 4]: SimdArray,
            Simd<[S; 4]>: $Op<Output = Simd<[S; 4]>>,
            Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
        {
            fn $op(self, other: Vector4<S>) -> $Output {
                let $lhs: Simd<[S; 4]> = self.into();
                let $rhs: Simd<[S; 4]> = other.into();
                $body
            }
        }

        impl<'a, S> $Op<&'a Vector4<S>> for Vector4<S>
        where
            S: BaseNum,
            [S; 4]: SimdArray,
            Simd<[S; 4]>: $Op<Output = Simd<[S; 4]>>,
            Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
        {
            fn $op(self, other: &'a Vector4<S>) -> $Output {
                let $lhs: Simd<[S; 4]> = self.into();
                let $rhs: Simd<[S; 4]> = (*other).into();
                $body
            }
        }

        impl<'a, S> $Op<Vector4<S>> for &'a Vector4<S>
        where
            S: BaseNum,
            [S; 4]: SimdArray,
            Simd<[S; 4]>: $Op<Output = Simd<[S; 4]>>,
            Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
        {
            fn $op(self, other: Vector4<S>) -> $Output {
                let $lhs: Simd<[S; 4]> = (*self).into();
                let $rhs: Simd<[S; 4]> = other.into();
                $body
            }
        }

        impl<'a, 'b, S> $Op<&'a Vector4<S>> for &'b Vector4<S>
        where
            S: BaseNum,
            [S; 4]: SimdArray,
            Simd<[S; 4]>: $Op<Output = Simd<[S; 4]>>,
            Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
        {
            fn $op(self, other: &'a Vector4<S>) -> $Output {
                let $lhs: Simd<[S; 4]> = (*self).into();
                let $rhs: Simd<[S; 4]> = (*other).into();
                $body
            }
        }
    };
}

impl_operator_simd2!(
    impl<S> Add<Vector4<S>> for Vector4<S> {
        fn add(lhs, rhs) -> Vector4<S> {
            (lhs + rhs).into()
        }
    }
);

impl_operator_simd2!(
    impl<S> Sub<Vector4<S>> for Vector4<S> {
        fn sub(lhs, rhs) -> Vector4<S> {
            (lhs - rhs).into()
        }
    }
);

impl_operator_simd2!(
    impl<S> Mul<S> for Vector4<S> {
        fn mul(lhs, rhs) -> Vector4<S> {
            (lhs * rhs).into()
        }
    }
);

impl_operator_simd2!(
    impl<S> Div<S> for Vector4<S> {
        fn div(lhs, rhs) -> Vector4<S> {
            (lhs / rhs).into()
        }
    }
);

impl<S> Neg for Vector4<S>
where
    S: Neg<Output = S>,
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Neg<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn neg(self) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        (-lhs).into()
    }
}

impl<S> AddAssign for Vector4<S>
where
    S: BaseNum,
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Add<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn add_assign(&mut self, other: Self) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = other.into();
        *self = (lhs + rhs).into();
    }
}

impl<S> SubAssign for Vector4<S>
where
    S: BaseNum,
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Sub<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn sub_assign(&mut self, other: Self) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = other.into();
        *self = (lhs - rhs).into();
    }
}

impl<S> MulAssign<S> for Vector4<S>
where
    S: BaseNum,
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Mul<S, Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn mul_assign(&mut self, other: S) {
        let lhs: Simd<[S; 4]> = (*self).into();
        *self = (lhs * other).into();
    }
}

// TODO(rodrigovalle):
//   $ cargo test --features simd -- --nocapture | grep optimized
// doesn't print "optmized impl"
// we should add a test for the *Assign operators
impl<S> DivAssign<S> for Vector4<S>
where
    S: BaseNum,
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Div<S, Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn div_assign(&mut self, other: S) {
        let lhs: Simd<[S; 4]> = (*self).into();
        *self = (lhs / other).into();
    }
}

impl<S> ElementWise for Vector4<S>
where
    S: BaseNum,
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Mul<Output = Simd<[S; 4]>> + Div<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn add_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        self + rhs
    }

    fn sub_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        self - rhs
    }

    fn mul_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        let rhs: Simd<[S; 4]> = rhs.into();
        (lhs * rhs).into()
    }

    fn div_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        let rhs: Simd<[S; 4]> = rhs.into();
        (lhs / rhs).into()
    }

    fn add_assign_element_wise(&mut self, rhs: Vector4<S>) {
        (*self) += rhs;
    }

    fn sub_assign_element_wise(&mut self, rhs: Vector4<S>) {
        (*self) -= rhs;
    }

    fn mul_assign_element_wise(&mut self, rhs: Vector4<S>) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = rhs.into();
        *self = (lhs * rhs).into()
    }

    fn div_assign_element_wise(&mut self, rhs: Vector4<S>) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = rhs.into();
        *self = (lhs / rhs).into()
    }
}

impl<S> ElementWise<S> for Vector4<S>
where
    S: BaseNum,
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Add<S, Output = Simd<[S; 4]>> + Sub<S, Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn add_element_wise(self, rhs: S) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        (lhs + rhs).into()
    }

    fn sub_element_wise(self, rhs: S) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        (lhs - rhs).into()
    }

    fn mul_element_wise(self, rhs: S) -> Vector4<S> {
        self * rhs
    }

    fn div_element_wise(self, rhs: S) -> Vector4<S> {
        self / rhs
    }

    fn add_assign_element_wise(&mut self, rhs: S) {
        let lhs: Simd<[S; 4]> = (*self).into();
        *self = (lhs + rhs).into();
    }

    fn sub_assign_element_wise(&mut self, rhs: S) {
        let lhs: Simd<[S; 4]> = (*self).into();
        *self = (lhs + rhs).into();
    }

    fn mul_assign_element_wise(&mut self, rhs: S) {
        (*self) *= rhs;
    }

    fn div_assign_element_wise(&mut self, rhs: S) {
        (*self) /= rhs;
    }
}
