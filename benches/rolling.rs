use criterion::{criterion_group, criterion_main, Criterion};
use tea_dyn::prelude::*;
const LENGTH: i32 = 10_000;

fn bench_rolling_apply_eager(c: &mut Criterion) {
    let data: Vec<_> = (0..LENGTH).collect();
    c.bench_function("rolling_apply eager", |b| {
        b.iter(|| {
            let _: Vec<_> = data
                .rolling_custom(100, |view| AggValidBasic::vsum(view.as_ref().titer()), None)
                .unwrap();
        })
    });
}

fn bench_rolling_apply_lazy(c: &mut Criterion) {
    let data: Vec<_> = (0..LENGTH).collect();
    let ctx = Context::new(data);
    let expr = s(0).rolling(100).apply(s(0).sum());
    c.bench_function("rolling_apply lazy", |b| {
        b.iter(|| expr.eval(&ctx, Some(Backend::Vec)))
    });
}

criterion_group!(benches, bench_rolling_apply_eager, bench_rolling_apply_lazy);
criterion_main!(benches);
