#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(let_chains)]
#![feature(vec_push_within_capacity)]
#![feature(vec_split_at_spare)]
#![feature(inherent_associated_types)]
#![feature(slice_pattern)]

extern crate alloc;

use core::{cmp::Ordering, hash::Hash, iter::Chain, ops::{AddAssign, SubAssign}, slice::SlicePattern};
use alloc::{alloc::{Allocator, Global}, boxed::Box, collections::{TryReserveError, VecDeque}, vec::Vec};

use num_traits::{Float, MulAddAssign, NumCast, Zero};

/// A similar data-structure to a [VecDeque](VecDeque), in that it is cyclic, but instead of cycling around its capacity, it cycles around its length. This makes rotation and SISO-behaviour simpler, but all operations that change the delay-line's length require it to be made contiguous first.
/// 
/// It's a more limited than [VecDeque](VecDeque), and only gives a miniscule performance boost for delay-line usage. For all other cases, you should probably just use [VecDeque](VecDeque).
/// 
/// # Examples
/// 
/// In this example, we mix in a delayed version of the signal `x`, delayed by 2 samples.
/// 
/// ```rust
/// use delay_line::DelayLine;
/// 
/// let mut x = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// 
/// let mut dl = DelayLine::new();
/// dl.resize(2);
/// 
/// for x in &mut x
/// {
///     *x += dl.delay(*x)*0.5;
/// }
/// 
/// assert_eq!(x, [1.0, 0.0, 0.5, 1.0, 0.0, 0.5])
/// ```
#[derive(Debug, Clone, Hash)]
pub struct DelayLine<T, A = Global>
where
    A: Allocator
{
    buffer: Vec<T, A>,
    offset: usize
}

pub type IntoIter<T, A = Global> = alloc::vec::IntoIter<T, A>;
pub type Iter<'a, T> = Chain<core::slice::Iter<'a, T>, core::slice::Iter<'a, T>>;
pub type IterMut<'a, T> = Chain<core::slice::IterMut<'a, T>, core::slice::IterMut<'a, T>>;

impl<T> DelayLine<T>
{
    pub fn new() -> Self
    {
        Self::new_in(Global)
    }
    pub fn with_capacity(capacity: usize) -> Self
    {
        Self::with_capacity_in(capacity, Global)
    }
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError>
    {
        Self::try_with_capacity_in(capacity, Global)
    }
}

impl<T, A> DelayLine<T, A>
where
    A: Allocator
{
    pub type IntoIter = IntoIter<T, A>;
    pub type Iter<'a> = Iter<'a, T>
    where
        T: 'a;
    pub type IterMut<'a> = IterMut<'a, T>
    where
        T: 'a;

    pub fn new_in(alloc: A) -> Self
    {
        Self {
            buffer: Vec::new_in(alloc),
            offset: 0
        }
    }
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self
    {
        Self {
            buffer: Vec::with_capacity_in(capacity, alloc),
            offset: 0
        }
    }
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError>
    {
        Ok(Self {
            buffer: Vec::try_with_capacity_in(capacity, alloc)?,
            offset: 0
        })
    }

    pub fn allocator(&self) -> &A
    {
        self.buffer.allocator()
    }
    pub fn is_empty(&self) -> bool
    {
        self.buffer.is_empty()
    }
    pub fn len(&self) -> usize
    {
        self.buffer.len()
    }
    pub fn capacity(&self) -> usize
    {
        self.buffer.capacity()
    }

    pub fn make_contiguous(&mut self) -> &mut Vec<T, A>
    {
        self.buffer.rotate_left(core::mem::replace(&mut self.offset, 0));
        &mut self.buffer
    }
    pub fn make_offset(&mut self, mut offset: usize)
    {
        let l = self.len();
        offset %= l;
        let doffset = (l + core::mem::replace(&mut self.offset, offset) - offset) % l;
        self.buffer.rotate_left(doffset);
    }

    pub fn reserve(&mut self, additional: usize)
    {
        self.buffer.reserve(additional);
    }
    pub fn reserve_exact(&mut self, additional: usize)
    {
        self.buffer.reserve_exact(additional);
    }
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>
    {
        self.buffer.try_reserve(additional)
    }
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError>
    {
        self.buffer.try_reserve_exact(additional)
    }

    pub fn shrink_to_fit(&mut self)
    {
        self.buffer.shrink_to_fit();
    }
    pub fn shrink_to(&mut self, min_capacity: usize)
    {
        self.buffer.shrink_to(min_capacity);
    }

    pub fn as_slices(&self) -> (&[T], &[T])
    {
        let (slice2, slice1) = unsafe {
            self.buffer.as_slice()
                .split_at_unchecked(self.offset)
        };
        (slice1, slice2)
    }
    pub fn as_mut_slices(&mut self) -> (&mut [T], &mut [T])
    {
        let (slice2, slice1) = unsafe {
            self.buffer.as_mut_slice()
                .split_at_mut_unchecked(self.offset)
        };
        (slice1, slice2)
    }
    pub fn leak<'a>(self) -> &'a mut [T]
    where
        A: 'a
    {
        Vec::from(self).leak()
    }
    pub fn into_boxed_slice(self) -> Box<[T], A>
    {
        Vec::from(self).into_boxed_slice()
    }

    pub fn into_iter(self) -> Self::IntoIter
    {
        Vec::from(self).into_iter()
    }
    pub fn iter(&self) -> Self::Iter<'_>
    {
        let (slice1, slice2) = self.as_slices();
        slice1.iter().chain(slice2.iter())
    }
    pub fn iter_mut(&mut self) -> Self::IterMut<'_>
    {
        let (slice1, slice2) = self.as_mut_slices();
        slice1.iter_mut().chain(slice2.iter_mut())
    }

    pub fn clear(&mut self) -> &mut Vec<T, A>
    {
        self.buffer.clear();
        self.offset = 0;
        &mut self.buffer
    }

    pub fn truncate(&mut self, len: usize)
    {
        if len == self.len()
        {
            return
        }
        self.make_contiguous().truncate(len);
    }
    pub fn resize(&mut self, len: usize)
    where
        T: Zero + Clone
    {
        if len == self.len()
        {
            return
        }
        self.make_contiguous().resize(len, Zero::zero());
    }

    pub fn insert(&mut self, i: usize, x: T)
    {
        let l = self.len() + 1;
        let j = i % l + self.offset;
        self.offset += (j >= l) as usize;
        self.buffer.insert(j % l, x)
    }
    pub fn remove(&mut self, i: usize) -> Option<T>
    {
        let l = self.len();
        if l == 0
        {
            return None;
        }
        let j = i % l + self.offset;
        self.offset += (j >= l) as usize;
        Some(self.buffer.remove(j % l))
    }

    pub fn push_in(&mut self, value: T)
    {
        self.make_contiguous().push(value)
    }
    pub fn push_in_within_capacity(&mut self, value: T) -> Result<(), T>
    {
        if self.len() == self.capacity()
        {
            return Err(value)
        }
        Ok(unsafe {
            self.make_contiguous().push_within_capacity(value).unwrap_unchecked()
        })
    }
    pub fn push_out(&mut self, value: T)
    {
        self.push_in(value);
        self.unrotate(1);
    }
    pub fn push_out_within_capacity(&mut self, value: T) -> Result<(), T>
    {
        let result = self.push_in_within_capacity(value);
        self.unrotate(result.is_ok() as usize);
        result
    }

    pub fn pop_in(&mut self) -> Option<T>
    {
        self.make_contiguous().pop()
    }
    pub fn pop_in_if(&mut self, predicate: impl FnOnce(&T) -> bool) -> Option<T>
    {
        self.pop_in_if_mut(|x| predicate(x))
    }
    pub fn pop_in_if_mut(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T>
    {
        let output = self.output_mut()?;
        if !predicate(output)
        {
            return None
        }
        unsafe {
            Some(self.pop_in().unwrap_unchecked())
        }
    }

    pub fn output(&self) -> Option<&T>
    {
        self.buffer.get(self.offset)
    }
    pub fn output_mut(&mut self) -> Option<&mut T>
    {
        self.buffer.get_mut(self.offset)
    }

    fn i_in(&self) -> Option<usize>
    {
        let l = self.len();
        let i = l.checked_sub(1)?;
        Some((i + self.offset) % l)
    }
    fn i(&self, i: usize) -> Option<usize>
    {
        let l = self.len();
        (i + self.offset).checked_rem(l)
    }

    pub fn input(&self) -> Option<&T>
    {
        let i = self.i_in()?;
        unsafe {
            Some(self.buffer.get_unchecked(i))
        }
    }
    pub fn input_mut(&mut self) -> Option<&mut T>
    {
        let i = self.i_in()?;
        unsafe {
            Some(self.buffer.get_unchecked_mut(i))
        }
    }

    pub fn get(&self, mut i: usize) -> Option<&T>
    {
        i = self.i(i)?;
        unsafe {
            Some(self.buffer.get_unchecked(i))
        }
    }
    pub fn get_mut(&mut self, mut i: usize) -> Option<&mut T>
    {
        i = self.i(i)?;
        unsafe {
            Some(self.buffer.get_unchecked_mut(i))
        }
    }

    pub fn rotate(&mut self, n: usize)
    {
        self.offset += n;
        self.offset %= self.len();
    }
    pub fn unrotate(&mut self, n: usize)
    {
        let l = self.len();
        self.rotate(l - n % l)
    }

    pub fn delay(&mut self, x: T) -> T
    {
        let y = match self.output_mut()
        {
            Some(out) => core::mem::replace(out, x),
            None => return x
        };
        self.rotate(1);
        y
    }
    pub fn delay_feedback(&mut self, x: T, feedback: T) -> T
    where
        T: Float + MulAddAssign
    {
        let y = match self.output_mut()
        {
            Some(out) => {
                let y = *out;
                out.mul_add_assign(feedback, x);
                y
            },
            None => return x
        };
        self.rotate(1);
        y
    }

    pub fn fill(&mut self, fill: T) -> &mut Vec<T, A>
    where
        T: Copy
    {
        self.buffer.fill(fill);
        self.offset = 0;
        &mut self.buffer
    }

    pub fn read_tap(&self, tap: T) -> Option<T>
    where
        T: Float
    {
        let l = self.len();
        l.checked_sub(1)
            .and_then(<T as NumCast>::from)
            .map(|ll| {
                let i = tap*ll;

                let p = i.fract();
                let q = T::one() - p;

                let read = |i: T, m: T| {
                    i.to_usize()
                        .and_then(|i| self.get(i))
                        .copied()
                        .map(|x| x*m)
                        .unwrap_or_else(T::zero)
                };
                
                let xq = read(i.floor(), q);
                let xp = read(i.ceil(), p);
                
                xq + xp
            })
    }

    fn rw_tap(&mut self, tap: T, dx: impl FnOnce(&mut Self, Option<usize>, Option<usize>, T, T) -> T, write: impl Fn(&mut T, T))
    where
        T: Float
    {
        let l = self.len();
        if let Some(ll) = l.checked_sub(1)
            .and_then(<T as NumCast>::from)
        {
            let i = tap*ll;

            let p = i.fract();
            let q = T::one() - p;
            
            let j0 = i.floor().to_usize();
            let j1 = i.ceil().to_usize();
            
            let dx = dx(self, j0, j1, q, p);

            let mut write = |j: Option<usize>, m: T| {
                if let Some(j) = j && let Some(dst) = self.get_mut(j)
                {
                    write(dst, dx*m)
                }
            };

            write(j0, q);
            write(j1, p);
        }
    }

    pub fn add_tap(&mut self, tap: T, x: T)
    where
        T: Float + AddAssign
    {
        self.rw_tap(
            tap,
            |_, _, _, _, _| {
                x
            },
            |y, dx| {
                *y += dx
            }
        );
    }

    pub fn sub_tap(&mut self, tap: T, x: T)
    where
        T: Float + SubAssign
    {
        self.rw_tap(
            tap,
            |_, _, _, _, _| {
                x
            },
            |y, dx| {
                *y -= dx
            }
        );
    }

    pub fn map_tap(&mut self, tap: T, x: impl FnOnce(T) -> T)
    where
        T: Float + AddAssign
    {
        self.rw_tap(
            tap,
            |this, j0, j1, q, p| {
                
                let read = |j: Option<usize>, m: T| {
                    j.and_then(|j| this.get(j).copied().map(|x| x*m))
                        .unwrap_or_else(T::zero)
                };

                let yq = read(j0, q);
                let yp = read(j1, p);

                let y = yq + yp;
                x(y) - y
            },
            |y, dx| {
                *y += dx
            }
        );
    }

    pub fn write_tap(&mut self, tap: T, x: T)
    where
        T: Float + AddAssign
    {
        self.map_tap(tap, |_| x);
    }

    pub fn stretch(&mut self, len: usize) -> &mut [T]
    where
        T: Float + AddAssign
    {
        let l0 = self.len();
        if len == l0
        {
            return &mut self.buffer;
        }

        let zero = T::zero();
        let one = T::one();
    
        let c = |i: usize, a: T| {
            T::from(i)
                .and_then(|i| {
                    let mut p = i*a;
                    if let (Some(j0), Some(j1)) = (p.floor().to_usize(), p.ceil().to_usize())
                    {
                        p = p.fract();
                        let q = one - p;
                
                        Some((j0, j1, q, p))
                    }
                    else
                    {
                        None
                    }
                })
        };
    
        self.make_contiguous();
        if len < l0
        {
            if len != 0
            {
                let a = T::from(len - 1).unwrap()/T::from(l0 - 1).unwrap();
                for i in 0..l0
                {
                    let mut x = core::mem::replace(unsafe {
                        self.buffer.get_unchecked_mut(i)
                    }, zero);

                    if let Some((j0, j1, q, p)) = c(i, a)
                    {
                        x = x*a;
                        unsafe {
                            *self.buffer.get_unchecked_mut(j0) += x*q;
                            *self.buffer.get_unchecked_mut(j1) += x*p;
                        }
                    }
                }
            }
            self.buffer.truncate(len);
        }
        else
        {
            self.buffer.resize(len, T::zero());
            if l0 != 0
            {
                let a = T::from(l0 - 1).unwrap()/T::from(len - 1).unwrap();
                for i in (0..len).rev()
                {
                    unsafe {
                        *self.buffer.get_unchecked_mut(i) = c(i, a).map(|(j0, j1, q, p)| {
                            *self.buffer.get_unchecked(j0)*q + *self.buffer.get_unchecked(j1)*p
                        }).unwrap_or(zero);
                    }
                }
            }
        }
        &mut self.buffer
    }
}

impl<T, A> Default for DelayLine<T, A>
where
    A: Allocator + Default
{
    fn default() -> Self
    {
        Self::new_in(Default::default())
    }
}

impl<T, A> From<Vec<T, A>> for DelayLine<T, A>
where
    A: Allocator
{
    fn from(buffer: Vec<T, A>) -> Self
    {
        Self {
            buffer,
            offset: 0
        }
    }
}
impl<T, A> From<VecDeque<T, A>> for DelayLine<T, A>
where
    A: Allocator
{
    fn from(value: VecDeque<T, A>) -> Self
    {
        Vec::from(value).into()
    }
}
impl<T, const N: usize> From<[T; N]> for DelayLine<T>
{
    fn from(value: [T; N]) -> Self
    {
        Vec::from(value).into()
    }
}

impl<T, A> From<DelayLine<T, A>> for Vec<T, A>
where
    A: Allocator
{
    fn from(mut delay: DelayLine<T, A>) -> Self
    {
        delay.make_contiguous();
        assert_eq!(delay.offset, 0);
        delay.buffer
    }
}
impl<T, A> From<DelayLine<T, A>> for VecDeque<T, A>
where
    A: Allocator
{
    fn from(delay: DelayLine<T, A>) -> Self
    {
        let offset = delay.offset;
        let mut deque = VecDeque::from(delay.buffer);
        deque.rotate_left(offset);
        deque
    }
}

impl<T> FromIterator<T> for DelayLine<T>
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self
    {
        Vec::from_iter(iter).into()
    }
}

impl<T, A> IntoIterator for DelayLine<T, A>
where
    A: Allocator
{
    type IntoIter = IntoIter<T, A>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter
    {
        self.into_iter()
    }
}
impl<'a, T, A> IntoIterator for &'a DelayLine<T, A>
where
    A: Allocator
{
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter
    {
        self.iter()
    }
}
impl<'a, T, A> IntoIterator for &'a mut DelayLine<T, A>
where
    A: Allocator
{
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter
    {
        self.iter_mut()
    }
}

fn eq_slices<T1, T2>((sa, sb): (&[T1], &[T1]), (oa, ob): (&[T2], &[T2])) -> bool
where
    T1: PartialEq<T2>
{
    match sa.len().cmp(&oa.len())
    {
        Ordering::Less => {
            let front = sa.len();
            let mid = oa.len() - front;

            let (oa_front, oa_mid) = oa.split_at(front);
            let (sb_mid, sb_back) = sb.split_at(mid);
            debug_assert_eq!(sa.len(), oa_front.len());
            debug_assert_eq!(sb_mid.len(), oa_mid.len());
            debug_assert_eq!(sb_back.len(), ob.len());
            sa == oa_front && sb_mid == oa_mid && sb_back == ob
        },
        Ordering::Equal => sa == oa && sb == ob,
        Ordering::Greater => {
            let front = oa.len();
            let mid = sa.len() - front;

            let (sa_front, sa_mid) = sa.split_at(front);
            let (ob_mid, ob_back) = ob.split_at(mid);
            debug_assert_eq!(sa_front.len(), oa.len());
            debug_assert_eq!(sa_mid.len(), ob_mid.len());
            debug_assert_eq!(sb.len(), ob_back.len());
            sa_front == oa && sa_mid == ob_mid && sb == ob_back
        },
    }
}

fn eq_slice<T1, T2>((sa, sb): (&[T1], &[T1]), o: &[T2]) -> bool
where
    T1: PartialEq<T2>
{
    let front = sa.len();

    let (oa, ob) = o.split_at(front);

    sa == oa && sb == ob
}

fn req_slice<T1, T2>(s: &[T1], (oa, ob): (&[T2], &[T2])) -> bool
where
    T1: PartialEq<T2>
{
    let front = oa.len();

    let (sa, sb) = s.split_at(front);

    sa == oa && sb == ob
}

macro_rules! impl_cmp_2_2 {
    ({$($g:tt)*} $a:ty, $b:ty $(where $($w:tt)*)?) => {
        impl<T1, T2, $($g)*> PartialEq<$b> for $a
        where
            T1: PartialEq<T2>,
            $($($w)*)?
        {
            fn eq(&self, other: &$b) -> bool
            {
                self.len() == other.len() && eq_slices(self.as_slices(), other.as_slices())
            }
        }
        impl<T1, T2, $($g)*> PartialOrd<$b> for $a
        where
            T1: PartialOrd<T2>,
            $($($w)*)?
        {
            fn partial_cmp(&self, other: &$b) -> Option<Ordering>
            {
                self.iter().partial_cmp(other.iter())
            }
        }
    };
    ({$($g:tt)*} $b:ty $(where $($w:tt)*)?) => {
        impl<T1, T2, A1, $($g)*> PartialEq<$b> for DelayLine<T1, A1>
        where
            T1: PartialEq<T2>,
            A1: Allocator,
            $($($w)*)?
        {
            fn eq(&self, other: &$b) -> bool
            {
                self.len() == other.len() && eq_slices(self.as_slices(), other.as_slices())
            }
        }
        impl<T1, T2, A1, $($g)*> PartialOrd<$b> for DelayLine<T1, A1>
        where
            T1: PartialOrd<T2>,
            A1: Allocator,
            $($($w)*)?
        {
            fn partial_cmp(&self, other: &$b) -> Option<Ordering>
            {
                self.iter().partial_cmp(other.iter())
            }
        }

        impl<T1, T2, A1, $($g)*> PartialEq<DelayLine<T1, A1>> for $b
        where
            T2: PartialEq<T1>,
            A1: Allocator,
            $($($w)*)?
        {
            fn eq(&self, other: &DelayLine<T1, A1>) -> bool
            {
                self.len() == other.len() && eq_slices(self.as_slices(), other.as_slices())
            }
        }
        impl<T1, T2, A1, $($g)*> PartialOrd<DelayLine<T1, A1>> for $b
        where
            T2: PartialOrd<T1>,
            A1: Allocator,
            $($($w)*)?
        {
            fn partial_cmp(&self, other: &DelayLine<T1, A1>) -> Option<Ordering>
            {
                self.iter().partial_cmp(other.iter())
            }
        }
    }
}

macro_rules! impl_cmp_2_1 {
    ({$($g:tt)*} $b:ty $(where $($w:tt)*)?) => {
        impl<T1, T2, A1, $($g)*> PartialEq<$b> for DelayLine<T1, A1>
        where
            T1: PartialEq<T2>,
            A1: Allocator,
            $($($w)*)?
        {
            fn eq(&self, other: &$b) -> bool
            {
                self.len() == other.len() && eq_slice(self.as_slices(), other.as_slice())
            }
        }
        impl<T1, T2, A1, $($g)*> PartialOrd<$b> for DelayLine<T1, A1>
        where
            T1: PartialOrd<T2>,
            A1: Allocator,
            $($($w)*)?
        {
            fn partial_cmp(&self, other: &$b) -> Option<Ordering>
            {
                self.iter().partial_cmp(other.iter())
            }
        }

        impl<T1, T2, A1, $($g)*> PartialEq<DelayLine<T1, A1>> for $b
        where
            T2: PartialEq<T1>,
            A1: Allocator,
            $($($w)*)?
        {
            fn eq(&self, other: &DelayLine<T1, A1>) -> bool
            {
                self.len() == other.len() && req_slice(self.as_slice(), other.as_slices())
            }
        }
        impl<T1, T2, A1, $($g)*> PartialOrd<DelayLine<T1, A1>> for $b
        where
            T2: PartialOrd<T1>,
            A1: Allocator,
            $($($w)*)?
        {
            fn partial_cmp(&self, other: &DelayLine<T1, A1>) -> Option<Ordering>
            {
                self.iter().partial_cmp(other.iter())
            }
        }
    };
}

type Array<T, const N: usize> = [T; N];
type Slice<T> = [T];

impl_cmp_2_2!({A1: Allocator, A2: Allocator} DelayLine<T1, A1>, DelayLine<T2, A2>);
impl_cmp_2_2!({A2: Allocator} VecDeque<T2, A2>);
impl_cmp_2_1!({A2: Allocator} Vec<T2, A2>);
impl_cmp_2_1!({const N: usize} Array<T2, N>);
impl_cmp_2_1!({} Slice<T2>);

impl<T, A> Eq for DelayLine<T, A>
where
    T: Eq,
    A: Allocator
{
    
}
impl<T, A> Ord for DelayLine<T, A>
where
    T: Ord,
    A: Allocator
{
    fn cmp(&self, other: &Self) -> Ordering
    {
        self.iter().cmp(other.iter())
    }
}

impl<T, A> FnOnce<(T,)> for DelayLine<T, A>
where
    A: Allocator
{
    type Output = T;

    extern "rust-call" fn call_once(mut self, (x,): (T,)) -> Self::Output
    {
        self.delay(x)
    }
}
impl<T, A> FnMut<(T,)> for DelayLine<T, A>
where
    A: Allocator
{
    extern "rust-call" fn call_mut(&mut self, (x,): (T,)) -> Self::Output
    {
        self.delay(x)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn it_works()
    {
        let mut x = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let mut dl = DelayLine::new();
        dl.resize(2);

        for x in &mut x
        {
            *x += dl.delay(*x);
            *dl.input_mut().unwrap() -= 0.1**x
        }

        println!("{:?}", x)
    }
}
