import { ClaimBlock } from '@/components/PageShell';
import RouteCards from '@/components/RouteCards';
import { SITE, SPATIAL_MAPS, STAT_TILES } from '@/lib/site';

export default function HomePage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-teal-700">
        {SITE.kicker}
      </p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">{SITE.title}</h1>
      <p className="mt-4 max-w-3xl text-lg text-slate-700">{SITE.lead}</p>

      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STAT_TILES.map((tile) => (
          <div
            key={tile.label}
            className="rounded-2xl border border-slate-200 bg-white/80 p-4 text-center"
          >
            <p className="text-2xl font-bold text-brand">{tile.value}</p>
            <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
              {tile.label}
            </p>
          </div>
        ))}
      </div>

      <section className="mt-10 rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Physical object
        </h2>
        <p className="mt-2 text-slate-800">{SITE.physicalObject}</p>
      </section>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {SPATIAL_MAPS.map((map) => (
          <figure
            key={map.id}
            className="overflow-hidden rounded-2xl border border-slate-200 bg-white/80"
          >
            <img src={map.src} alt={map.title} className="mx-auto max-h-48 w-auto bg-slate-50 p-2" />
            <figcaption className="p-4">
              <p className="text-sm font-semibold text-slate-900">{map.title}</p>
              <p className="mt-1 text-xs text-slate-600">{map.caption}</p>
            </figcaption>
          </figure>
        ))}
      </div>

      <div className="mt-8">
        <ClaimBlock />
      </div>

      <section className="mt-10 rounded-2xl border border-amber-200 bg-amber-50/60 p-5 text-sm text-slate-700">
        <h2 className="font-semibold text-slate-900">What this site is not</h2>
        <p className="mt-2">
          Not a SOTA page for Tessera-the-method. Not a frozen-PDF screenshot. Production A–D TikZ
          layouts are not published here. ρ = +1.00 is saturated Spearman, not an effect size.
        </p>
      </section>

      <section className="mt-10">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Explore
        </h2>
        <RouteCards />
      </section>
    </div>
  );
}
