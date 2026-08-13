import PageShell from '@/components/PageShell';
import { SPATIAL_MAPS } from '@/lib/site';

export default function ResultsPage() {
  return (
    <PageShell title="Results" kicker="Geometry first">
      <p>
        Domain geometry and adjacency on tissue. Raster tiles from the synthetic 32×32 tessellation —
        not unbaked production TikZ A–D layouts.
      </p>
      <div className="grid gap-6 sm:grid-cols-2">
        {SPATIAL_MAPS.map((map) => (
          <figure
            key={map.id}
            className="overflow-hidden rounded-2xl border border-slate-200 bg-white/80"
          >
            <img src={map.src} alt={map.title} className="mx-auto bg-slate-50 p-2" />
            <figcaption className="p-4">
              <p className="text-sm font-semibold text-slate-900">{map.title}</p>
              <p className="mt-1 text-sm text-slate-600">{map.caption}</p>
            </figcaption>
          </figure>
        ))}
      </div>
      <section className="rounded-2xl border border-slate-200 bg-white/80 p-5">
        <h2 className="text-lg font-semibold text-slate-900">Coordinate-shuffle intervention</h2>
        <p className="mt-2 text-sm text-slate-600">
          Scramble cell positions while holding expression and domain labels fixed. Spatial-prior
          advantage rises with resulting contiguity and flips sign — a physical intervention on
          geometry, not an ARI tweak.
        </p>
      </section>
    </PageShell>
  );
}
