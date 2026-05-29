/* Cytoscape setup for the graph view.
   The Jinja template injects `#graph-data` as a JSON script tag with
   `{nodes: [...], edges: [...]}` shape. */

(function () {
  'use strict';

  const dataEl = document.getElementById('graph-data');
  if (!dataEl) return;

  let data;
  try {
    data = JSON.parse(dataEl.textContent);
  } catch (_e) {
    return;
  }
  if (!data || !data.nodes) return;

  const container = document.getElementById('cy');
  if (!container) return;

  // Degrade quietly if the (vendored) cytoscape bundle didn't load, instead of
  // throwing a ReferenceError and leaving a blank canvas with no explanation.
  if (typeof window.cytoscape !== 'function') {
    container.innerHTML =
      '<div class="graph-unavailable">graph renderer unavailable — run <code>scripts/vendor-frontend.sh</code></div>';
    return;
  }

  const elements = [
    ...data.nodes.map(n => ({
      data: n,
      classes: n.kind === 'center' ? 'center' : 'neighbor',
    })),
    ...data.edges.map(e => ({ data: e })),
  ];

  const cy = window.cytoscape({
    container,
    elements,
    layout: {
      name: 'concentric',
      concentric: function (node) {
        return node.data('kind') === 'center' ? 2 : 1;
      },
      levelWidth: function () { return 1; },
      minNodeSpacing: 72,
      avoidOverlap: true,
      animate: false,
      padding: 32,
    },
    wheelSensitivity: 0.2,
    minZoom: 0.3,
    maxZoom: 4,
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'rgb(39 39 42)',
          'border-color': 'rgb(82 82 91)',
          'border-width': 1,
          'label': 'data(title)',
          'color': 'rgb(212 212 216)',
          'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
          'font-size': 10,
          'text-valign': 'bottom',
          'text-halign': 'center',
          'text-margin-y': 10,
          'text-wrap': 'wrap',
          'text-max-width': 140,
          'width': 34,
          'height': 34,
        },
      },
      {
        selector: 'node.center',
        style: {
          'background-color': 'rgb(30 58 138)',     /* blue-900 */
          'border-color': 'rgb(37 99 235)',         /* blue-600 */
          'border-width': 2,
          'shape': 'round-rectangle',
          'width': 56,
          'height': 56,
          'color': 'rgb(219 234 254)',              /* blue-100 */
          'font-weight': 600,
          'text-max-width': 180,
        },
      },
      {
        selector: 'node:selected',
        style: {
          'border-color': 'rgb(96 165 250)',        /* blue-400 */
          'border-width': 3,
          'color': 'rgb(244 244 245)',              /* zinc-100 */
        },
      },
      {
        selector: 'node:active',
        style: {
          'overlay-opacity': 0,
        },
      },
      {
        selector: 'edge',
        style: {
          'line-color': 'rgb(63 63 70)',
          'width': 1,
          'curve-style': 'bezier',
          'label': 'data(label)',
          'font-size': 9,
          'color': 'rgb(113 113 122)',
          'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
          'text-rotation': 'autorotate',
          'text-background-color': 'rgb(9 9 11)',
          'text-background-opacity': 1,
          'text-background-padding': 3,
          'target-arrow-shape': 'none',
        },
      },
      {
        selector: 'edge:selected',
        style: {
          'line-color': 'rgb(96 165 250)',
          'width': 1.5,
          'color': 'rgb(212 212 216)',
        },
      },
    ],
  });

  // Pre-select the center node so the inspector renders something useful
  // on first paint without requiring a click.
  const centerNode = cy.nodes('.center').first();
  if (centerNode.length) {
    centerNode.select();
    renderInspector(centerNode.data(), incidentEdges(centerNode));
  }

  cy.on('tap', 'node', evt => {
    const n = evt.target;
    renderInspector(n.data(), incidentEdges(n));
  });

  cy.on('tap', function (evt) {
    if (evt.target === cy) {
      // Tapped on empty canvas — reset to the center.
      const c = cy.nodes('.center').first();
      if (c.length) {
        cy.elements().unselect();
        c.select();
        renderInspector(c.data(), incidentEdges(c));
      }
    }
  });

  function incidentEdges(node) {
    return node.connectedEdges().map(e => {
      const src = e.source().data('title');
      const tgt = e.target().data('title');
      return {
        label: e.data('label'),
        other: e.source().id() === node.id() ? tgt : src,
        other_id: e.source().id() === node.id() ? e.target().id() : e.source().id(),
      };
    });
  }

  function renderInspector(node, edges) {
    const body = document.getElementById('inspector-body');
    if (!body) return;
    const kind = node.kind === 'center' ? 'this doc' : 'neighbor';
    const kindClass = node.kind === 'center' ? '' : 'neighbor';
    const safeTitle = escapeHtml(node.title || node.id);
    const safeId = escapeHtml(node.id);

    let edgesHtml = '';
    if (edges.length) {
      edgesHtml = `
        <div class="tick-rule"></div>
        <div class="uppercase tracking-widest text-zinc-400 mb-2" style="font-size:0.625rem">
          incident edges (${edges.length})
        </div>
        <ul class="space-y-1.5">
          ${edges.map(e => `
            <li class="text-xs">
              <span class="font-mono text-zinc-400">${escapeHtml(e.label || '—')}</span>
              <span class="text-zinc-400">→</span>
              <a href="/documents/${encodeURIComponent(e.other_id)}"
                 class="text-zinc-300 hover:text-blue-300">${escapeHtml(e.other)}</a>
            </li>
          `).join('')}
        </ul>
      `;
    } else {
      edgesHtml = `
        <div class="tick-rule"></div>
        <p class="text-xs text-zinc-400 italic">no incident edges.</p>
      `;
    }

    body.innerHTML = `
      <div class="mb-2">
        <span class="inspector-kind ${kindClass}">${kind}</span>
      </div>
      <h2 class="text-zinc-100 text-sm font-medium leading-tight mb-3 break-words">${safeTitle}</h2>
      <dl>
        <div class="dl-row"><dt>doc_id</dt><dd>${safeId}</dd></div>
        ${node.kind !== 'center'
          ? `<div class="dl-row"><dt>relation</dt><dd style="font-family:ui-sans-serif,system-ui,sans-serif;color:rgb(161 161 170)">shares entity</dd></div>`
          : ''}
      </dl>
      ${node.kind !== 'center'
        ? `<a href="/documents/${encodeURIComponent(safeId)}"
              class="mt-4 inline-block text-xs text-blue-400 hover:text-blue-300">open document →</a>`
        : `<a href="/graph/${encodeURIComponent(safeId)}"
              class="mt-4 inline-block text-xs text-zinc-400 hover:text-zinc-300">re-center on this →</a>`
      }
      ${edgesHtml}
    `;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  // Handle window resize so the canvas re-fits.
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => cy.resize().fit(undefined, 48), 100);
  });
})();
