import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function ClaimsPage() {
  return (
    <PageShell title="Claims" kicker="Falsifiable statements">
      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Primary claim</h2>
        <p className="mt-3 text-slate-700">{SITE.primaryClaim}</p>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Would refute
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Spatial prior helps equally on scattered-niche tissues (CODEX adjacency 0.022)</li>
          <li>Shuffle intervention leaves spatial-prior advantage flat across contiguity</li>
          <li>Label-free coh_gain proxy survives search correction as a significant predictor</li>
        </ul>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Out of scope
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Tessera-the-method as universal SOTA across the eleven-platform panel</li>
          <li>ρ = +1.00 interpreted as effect size rather than saturated rank correlation</li>
          <li>Journal venue packaging, BibTeX stubs, or invented article DOI</li>
        </ul>
      </section>
    </PageShell>
  );
}
