import PageShell from '@/components/PageShell';
import { ADJACENCY_ROWS } from '@/lib/site';

export default function EvidencePage() {
  return (
    <PageShell title="Evidence" kicker="Metrics and observational law">
      <div className="grid gap-4 sm:grid-cols-3">
        {[
          { value: '+0.72', label: 'Observational Spearman', note: 'p = 0.012 · 11 platforms' },
          { value: '+1.00', label: 'Synthetic ρ (saturated)', note: 'Sign flip load-bearing' },
          { value: '7', label: 'Distinct winners', note: 'No universal method crown' },
        ].map((tile) => (
          <div
            key={tile.label}
            className="rounded-2xl border border-slate-200 bg-white/80 p-5 text-center"
          >
            <p className="text-2xl font-bold text-brand">{tile.value}</p>
            <p className="mt-1 text-sm font-semibold text-slate-800">{tile.label}</p>
            <p className="mt-1 text-xs text-slate-500">{tile.note}</p>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white/80">
        <table className="min-w-full text-left text-sm">
          <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-3">Platform</th>
              <th className="px-4 py-3">Contiguity</th>
              <th className="px-4 py-3">Geometry</th>
              <th className="px-4 py-3">Winner ARI</th>
              <th className="px-4 py-3">Spatial adv.</th>
            </tr>
          </thead>
          <tbody>
            {ADJACENCY_ROWS.map((row) => (
              <tr key={row.platform} className="border-b border-slate-100 last:border-0">
                <td className="px-4 py-2 font-medium text-slate-800">{row.platform}</td>
                <td className="px-4 py-2 font-mono text-slate-600">{row.contiguity}</td>
                <td className="px-4 py-2 text-slate-600">{row.geometry}</td>
                <td className="px-4 py-2 font-mono text-slate-600">{row.winnerAri}</td>
                <td className="px-4 py-2 font-mono text-slate-600">{row.spatialAdv}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-500">
        Contiguity = fraction of spatial kNN sharing ground-truth label. Winner ARI is best-config
        mean over three seeds.
      </p>
    </PageShell>
  );
}
