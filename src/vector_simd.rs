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

use vector::*;

use structure::*;

use std::mem;
use std::ops::*;

use num::BaseNum;
use packed_simd::{Simd, SimdArray};

impl<S> From<Simd<[S; 4]>> for Vector4<S>
where
    S: Copy,
    [S; 4]: SimdArray + From<Simd<[S; 4]>>
{
    fn from(f: Simd<[S; 4]>) -> Self {
        let arr: [S; 4] = f.into();
        Vector4 { x: arr[0], y: arr[1], z: arr[2], w: arr[3] }
    }
}

impl<S> Into<Simd<[S; 4]>> for Vector4<S>
where
    [S; 4]: SimdArray + Into<Simd<[S; 4]>>
{
    fn into(self) -> Simd<[S; 4]> {
        let arr: [S; 4] = [self.x, self.y, self.w, self.z];
        arr.into()
    }
}

// TODO: hard to replicate without grouping SIMD types under traits that
// implement e.g. sqrt(), approx_rsqrt(), and approx_reciprocal()
// might need to make some custom traits to expose these methods generically
impl Vector4<f32> {
    /// Compute and return the square root of each element.
    #[inline]
    pub fn sqrt_element_wide(self) -> Self {
        let s: Simd<[f32; 4]> = self.into();
        s.sqrt().into()
    }

    /// Compute and return the reciprocal of the square root of each element.
    #[inline]
    pub fn rsqrt_element_wide(self) -> Self {
        let s: Simdf32x4 = self.into();
        s.approx_rsqrt().into()
    }

    /// Compute and return the reciprocal of each element.
    #[inline]
    pub fn recip_element_wide(self) -> Self {
        let s: Simdf32x4 = self.into();
        s.approx_reciprocal().into()
    }
}

impl<S> Add<Vector4<S>> for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Add<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    type Output = Self;

    fn add(self, other: Vector4<S>) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        let rhs: Simd<[S; 4]> = other.into();
        (lhs + rhs).into()
    }
}

impl<S> Sub<Vector4<S>> for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Sub<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    type Output = Self;

    fn sub(self, other: Vector4<S>) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        let rhs: Simd<[S; 4]> = other.into();
        (lhs - rhs).into()
    }
}

impl<S> Mul<S> for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Mul<S, Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    type Output = Self;

    fn mul(self, other: S) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        (lhs * other).into()
    }
}

impl<S> Div<S> for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Div<S, Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    type Output = Self;

    fn div(self, other: S) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        (lhs / other).into()
    }
}

impl<S> Neg for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Neg<Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    type Output = Self;

    fn neg(self) -> Vector4<S> {
        let lhs: Simd<[S; 4]> = self.into();
        (-lhs).into()
    }
}

impl<S> AddAssign for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Add<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>,
{
    fn add_assign(&mut self, rhs: Self) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = rhs.into();
        *self = (s + rhs).into();
    }
}

impl<S> SubAssign for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Sub<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>
{
    fn sub_assign(&mut self, rhs: Self) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = rhs.into();
        *self = (s - rhs).into();
    }
}

// dont implement SubAssign<S> for Vector4<S> because that would be adding a
// scalar to a vector which is not well defined

impl<S> MulAssign for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Mul<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>
{
    fn mul_assign(&mut self, other: Vector4<S>) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = rhs.into();
        *self = (s * rhs).into();
    }
}

impl<S> DivAssign for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Div<Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>
{
    fn div_assign(&mut self, other: Vector4<S>) {
        let lhs: Simd<[S; 4]> = (*self).into();
        let rhs: Simd<[S; 4]> = rhs.into();
        *self = (s / rhs).into();
    }
}

impl<S> MulAssign<S> for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Mul<S, Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>
{
    fn mul_assign(&mut self, other: S) {
        let lhs: Simd<[S; 4]> = (*self).into();
        *self = (lhs * other).into();
    }
}

impl<S> DivAssign<S> for Vector4<S>
where
    [S; 4]: SimdArray,
    Simd<[S; 4]>: Div<S, Output = Simd<[S; 4]>>,
    Vector4<S>: From<Simd<[S; 4]>> + Into<Simd<[S; 4]>>
{
    fn div_assign(&mut self, other: S) {
        let lhs: Simd<[S; 4]> = (*self).into();
        *self = (s / other).into();
    }
}

impl<S> ElementWise for Vector4<S>
where
    //S: ops::Mul<Output = S>,
    S: BaseNum,
    [S; 4]: SimdArray,
    Vector4<S>: Into<Simd<[S; 4]>> + From<Simd<[S; 4]>>,
    //Simd<[S; 4]>: ops::Mul<Output = Simd<[S; 4]>>,
    Simd<[S; 4]>: BaseNum,
{
    fn add_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        self + rhs
    }

    fn sub_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        self - rhs
    }

    fn mul_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        self * rhs
    }

    fn div_element_wise(self, rhs: Vector4<S>) -> Vector4<S> {
        self / rhs
    }

    fn add_assign_element_wise(&mut self, rhs: Vector4<S>) {
        (*self) += rhs;
    }

    fn mul_assign_element_wise(&mut self, rhs: Vector4<S>) {
        (*self) *= rhs;
    }

    fn div_assign_element_wise(&mut self, rhs: Vector4<S>) {
        (*self) /= rhs;
    }
}

impl<S> ElementWise<S> for Vector4<S> {
    fn add_element_wise(self, rhs: S) -> Vector4<S> {
        unimplemented!();
    }

    fn sub_element_wise(self, rhs: S) -> Vector4<S> {
        unimplemented!();
    }

    fn mul_element_wise(self, rhs: S) -> Vector4<S> {
        self * rhs
    }

    fn div_element_wise(self, rhs: S) -> Vector4<S> {
        self / rhs
    }

    fn add_assign_element_wise(&mut self, rhs: S) {
        unimplemented!();
    }

    #[inline]
    fn sub_assign_element_wise(&mut self, rhs: S) {
        unimplemented!();
    }

    fn mul_assign_element_wise(&mut self, rhs: S) {
        (*self) *= rhs;
    }

    fn div_assign_element_wise(&mut self, rhs: S) {
        (*self) /= rhs;
    }
}
