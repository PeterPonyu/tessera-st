import PageShell from '@/components/PageShell';

export default function MethodsPage() {
  return (
    <PageShell title="Methods" kicker="Protocol and panel">
      <p>
        Primary synthetic sweep on a 32×32 lattice with four stripe domains and a disc niche.
        Observational panel spans eleven technologically diverse platforms.
      </p>
      <ul className="list-disc space-y-2 pl-5">
        <li>
          <strong>Contiguity:</strong> fraction of spatial kNN neighbours (
          <span className="font-mono text-sm">k = 6</span>) sharing the ground-truth domain label.
        </li>
        <li>
          <strong>Intervention:</strong> coordinate shuffle with expression and labels held fixed;
          decoupled v2 Arm B manipulates contiguity alone.
        </li>
        <li>
          <strong>Backends:</strong> non-spatial floor, neighbour-mean smoother, and author-recommended
          clustering backends — no universal method winner assumed.
        </li>
        <li>
          <strong>Exclusions:</strong> no frozen-PDF rebuild, no production TikZ A–D on this leaf.
        </li>
      </ul>
    </PageShell>
  );
}
