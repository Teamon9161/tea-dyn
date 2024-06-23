#![feature(test)]

extern crate test;
use test::Bencher;

use tea_dyn::prelude::*;
// use tevec::prelude::*;

// const LENGTH: i32 = 10_000_000;
const LENGTH: i32 = 10_000;

#[bench]
fn bench_rolling_apply_eager(b: &mut Bencher) {
    let data: Vec<_> = (0..LENGTH).collect();
    b.iter(|| {
        let _: Vec<_> = data
            .rolling_custom(100, |view| AggValidBasic::vsum(view.as_ref().titer()), None)
            .unwrap();
    });
}

#[bench]
fn bench_rolling_apply_lazy(b: &mut Bencher) {
    let data: Vec<_> = (0..LENGTH).collect();
    let ctx = Context::new(data);
    let expr = s(0).rolling(100).apply(s(0).sum());
    b.iter(|| expr.eval(&ctx, Some(Backend::Vec)));
}
