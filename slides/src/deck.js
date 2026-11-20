(() => {
  const stage = document.querySelector('#stage');
  const slides = [...stage.querySelectorAll(':scope > .slide')];
  const count = document.querySelector('#count');
  const progress = document.querySelector('#progress i');
  const overview = document.querySelector('#overview');
  const notes = document.querySelector('#notes');
  const noteBody = document.querySelector('#note-body');
  const photos = JSON.parse(document.querySelector('#photo-data').textContent);
  const jumpButtons = [];
  let index = 0;
  let touchStart = null;

  function setPhoto(img, key) {
    img.src = photos[key];
    img.dataset.photo = key;
  }

  document.querySelectorAll('[data-photo]').forEach(img => setPhoto(img, img.dataset.photo));
  document.querySelectorAll('[data-atlas]').forEach(atlas => {
    const detail = atlas.querySelector('.feature-detail');
    const buttons = [...atlas.querySelectorAll('[data-sample]')];
    buttons.forEach(button => button.addEventListener('click', () => {
      const id = button.dataset.sample;
      buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      detail.querySelector('[data-triplet-id]').textContent = 'Feather ' + id;
      detail.querySelectorAll('[data-role]').forEach(img => {
        setPhoto(img, id + '-' + img.dataset.role);
        img.alt = img.dataset.alt + ' for Feather source ' + id;
      });
    }));
  });

  function setupReplay(panel) {
    const record = JSON.parse(panel.querySelector('[data-diagram="generator"]').textContent);
    const button = panel.querySelector('[data-gen-play]');
    const slider = panel.querySelector('[data-gen-scrub]');
    const input = panel.querySelector('[data-gen-input]');
    const maps = [...panel.querySelectorAll('[data-gen-map]')];
    const loss = panel.querySelector('[data-gen-value]');
    const counter = panel.querySelector('[data-gen-step]');
    const dot = panel.querySelector('[data-gen-dot]');
    const last = record.frames.length - 1;
    let current = 0;
    let timer;

    function showFrame(frameIndex) {
      current = frameIndex;
      const frame = record.frames[current];
      slider.value = current;
      slider.setAttribute('aria-valuetext', 'Step ' + frame.step + ' of ' + record.steps);
      setPhoto(input, frame.input);
      input.alt = 'Candidate image at optimization step ' + frame.step;
      maps.forEach((img, index) => setPhoto(img, frame.maps[index]));
      loss.textContent = frame.error.toFixed(3);
      counter.textContent = 'Step ' + frame.step + ' / ' + record.steps;
      const [x, y] = record.points[current];
      dot.setAttribute('cx', x);
      dot.setAttribute('cy', y);
    }

    function stop() {
      window.clearInterval(timer);
      timer = undefined;
      panel.classList.remove('is-playing');
      button.textContent = current > 0 && current < last ? '▶ Continue' : '▶ Replay optimization';
    }

    button.addEventListener('click', () => {
      if (timer !== undefined) { stop(); return; }
      if (current === last) showFrame(0);
      panel.classList.add('is-playing');
      button.textContent = 'Ⅱ Pause';
      timer = window.setInterval(() => {
        showFrame(current + 1);
        if (current === last) stop();
      }, 340);
    });
    slider.addEventListener('input', () => {
      showFrame(Number(slider.value));
      stop();
    });
    panel.addEventListener('stop-replay', stop);
    showFrame(0);
  }

  document.querySelectorAll('[data-generator]').forEach(setupReplay);

  document.querySelectorAll('[data-observer]').forEach(probe => {
    const roles = [...probe.querySelectorAll('[data-observer-role]')];
    const channels = [...probe.querySelectorAll('[data-observer-channel]')];
    let role = roles[0];
    let channel = channels[0];

    function updateProbe() {
      const condition = role.dataset.observerRole;
      const id = channel.dataset.observerChannel;
      const label = role.textContent.trim();
      roles.forEach(button => button.setAttribute('aria-pressed', String(button === role)));
      channels.forEach(button => button.setAttribute('aria-pressed', String(button === channel)));
      const input = probe.querySelector('[data-observer-input]');
      const map = probe.querySelector('[data-observer-map]');
      const vector = probe.querySelector('[data-observer-vector]');
      setPhoto(input, '0000-' + condition);
      input.alt = label + ', Feather source 0000';
      setPhoto(map, 'observer-' + condition + '-' + id);
      map.alt = label + ' activation map, Stage 3 channel ' + id;
      setPhoto(vector, 'observer-' + condition + '-pooled');
      vector.alt = label + ' pooled channel means arranged in a display grid';
    }

    roles.forEach(button => button.addEventListener('click', () => { role = button; updateProbe(); }));
    channels.forEach(button => button.addEventListener('click', () => { channel = button; updateProbe(); }));
  });

  document.querySelectorAll('[data-topk]').forEach(demo => {
    const range = demo.querySelector('[data-gate-count]');
    const ranked = [...demo.querySelectorAll('[data-gate-score]')].sort(
      (a, b) => Number(b.dataset.gateScore) - Number(a.dataset.gateScore)
    );
    range.addEventListener('input', () => {
      const keep = Number(range.value);
      ranked.forEach((bar, index) => bar.classList.toggle('keeps', index < keep));
      demo.querySelector('[data-gate-value]').textContent = keep;
      demo.querySelector('[data-gate-summary]').textContent = keep + ' of ' + ranked.length + ' retained';
    });
  });

  document.querySelectorAll('[data-switcher]').forEach(switcher => {
    const buttons = [...switcher.querySelectorAll('[data-panel-button]')];
    const panels = [...switcher.querySelectorAll('[data-panel]')];
    buttons.forEach(button => button.addEventListener('click', () => {
      const selected = button.dataset.panelButton;
      buttons.forEach(option => option.setAttribute('aria-pressed', String(option === button)));
      panels.forEach(panel => { panel.hidden = panel.dataset.panel !== selected; });
    }));
  });

  document.querySelectorAll('[data-reconstruction-lab]').forEach(lab => {
    const { cases } = JSON.parse(lab.querySelector('[data-diagram="reconstruction"]').textContent);
    const selectors = [
      [...lab.querySelectorAll('[data-recon-case]')],
      [...lab.querySelectorAll('[data-recon-pca]')],
      [...lab.querySelectorAll('[data-recon-view]')]
    ];
    const [caseButtons, pcaButtons, viewButtons] = selectors;
    const labels = {source: 'natural source', standard: 'standard metamer', adversarial: 'adversarial metamer'};
    let caseId = Object.keys(cases)[0];
    let method = 'pca16';
    let residual = false;

    function refresh() {
      const componentLabel = method === 'pca16' ? '16 components' : '64 components';
      lab.querySelector('[data-recon-pca-label]').textContent = componentLabel;
      lab.querySelector('[data-recon-pca-heading]').textContent = residual ? 'PCA absolute error' : 'PCA reconstruction';
      lab.querySelector('[data-recon-sae-heading]').textContent = residual ? 'TopK SAE absolute error' : 'TopK SAE reconstruction';
      lab.querySelector('[data-recon-error-scale]').hidden = !residual;
      caseButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.reconCase === caseId)));
      pcaButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.reconPca === method)));
      viewButtons.forEach(button => button.setAttribute('aria-pressed', String((button.dataset.reconView === 'residual') === residual)));

      lab.querySelectorAll('[data-recon-input]').forEach(img => {
        const role = img.dataset.reconInput;
        setPhoto(img, caseId + '-' + role);
        img.alt = labels[role] + ' in Feather triplet ' + caseId;
      });
      lab.querySelectorAll('[data-recon-map]').forEach(img => {
        const role = img.dataset.reconRole;
        const panel = img.dataset.reconMap;
        const model = panel === 'pca16' ? method : panel;
        const errorView = residual && model !== 'raw';
        const key = 'reconstruction-' + caseId + '-' + role + '-' + model;
        const name = model === 'raw' ? 'Original activation' : model === 'sae' ? 'TopK SAE' : 'PCA with ' + componentLabel;
        setPhoto(img, key + (errorView ? '-residual' : ''));
        img.alt = name + (errorView ? ' absolute residual' : '') + ', ' + labels[role] + ' in triplet ' + caseId;
        if (model === 'raw') return;
        const stat = lab.querySelector('[data-recon-stat="' + role + '-' + panel + '"]');
        stat.textContent = 'MSE ' + cases[caseId][role][model].toFixed(4);
      });
    }

    caseButtons.forEach(button => button.addEventListener('click', () => {
      caseId = button.dataset.reconCase;
      refresh();
    }));
    pcaButtons.forEach(button => button.addEventListener('click', () => {
      method = button.dataset.reconPca;
      refresh();
    }));
    viewButtons.forEach(button => button.addEventListener('click', () => {
      residual = button.dataset.reconView === 'residual';
      refresh();
    }));
  });

  function fit() {
    const scale = Math.min(innerWidth / 1440, innerHeight / 810);
    stage.style.setProperty('--scale', scale);
  }

  function renderNotes() {
    document.querySelector('#note-title').textContent = slides[index].dataset.title;
    noteBody.replaceChildren(slides[index].querySelector('template')?.content.cloneNode(true) ?? '');
  }

  function show(next, updateHash = true) {
    const previous = index;
    index = Math.min(slides.length - 1, Math.max(0, next));
    if (index !== previous) {
      slides[previous].querySelector('[data-generator].is-playing')?.dispatchEvent(new Event('stop-replay'));
    }
    slides.forEach((slide, i) => {
      slide.classList.toggle('active', i === index);
      slide.hidden = i !== index;
      jumpButtons[i].setAttribute('aria-current', String(i === index));
    });
    count.textContent = (index + 1) + ' / ' + slides.length;
    progress.style.width = (100 * (index + 1) / slides.length) + '%';
    document.querySelector('[data-action=prev]').disabled = index === 0;
    document.querySelector('[data-action=next]').disabled = index === slides.length - 1;
    if (notes.open) renderNotes();
    if (updateHash && location.hash !== '#' + slides[index].id) {
      try { history.replaceState(null, '', '#' + slides[index].id); }
      catch { location.hash = slides[index].id; }
    }
  }

  function fromHash() {
    const hash = location.hash.slice(1);
    const direct = slides.findIndex(slide => slide.id === hash);
    const ordinal = /^slide-(\d+)$/.exec(hash);
    return direct >= 0 ? direct : ordinal ? Number(ordinal[1]) - 1 : 0;
  }

  function fullscreen() {
    if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
    else document.documentElement.requestFullscreen?.().catch(() => {});
  }

  const actions = {
    prev: () => show(index - 1),
    next: () => show(index + 1),
    overview: () => overview.open ? overview.close() : overview.showModal(),
    notes: () => {
      if (notes.open) notes.close();
      else { renderNotes(); notes.showModal(); }
    },
    fullscreen
  };

  slides.forEach((slide, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = (i + 1) + '  ' + slide.dataset.title;
    button.addEventListener('click', () => { overview.close(); show(i); });
    document.querySelector('#overview-list').append(button);
    jumpButtons.push(button);
  });

  document.querySelectorAll('[data-action]').forEach(button => {
    button.addEventListener('click', () => actions[button.dataset.action]());
  });
  window.addEventListener('resize', fit);
  window.addEventListener('hashchange', () => show(fromHash(), false));
  window.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.repeat) return;
    if (notes.open || overview.open) {
      if ((event.key.toLowerCase() === 'n' && notes.open)
        || (event.key.toLowerCase() === 'o' && overview.open)) {
        (notes.open ? notes : overview).close();
      }
      return;
    }
    const key = event.key.toLowerCase();
    if (event.target.closest('input, select, textarea, [contenteditable]')) return;
    if (key === ' ' && event.target.closest('button, a')) return;
    const action = {
      arrowleft: actions.prev, pageup: actions.prev, arrowright: actions.next,
      pagedown: actions.next, ' ': actions.next, home: () => show(0),
      end: () => show(slides.length - 1), o: actions.overview,
      n: actions.notes, f: actions.fullscreen
    }[key];
    if (action) { event.preventDefault(); action(); }
  });
  stage.addEventListener('touchstart', event => {
    touchStart = event.target.closest('button, a') ? null : event.changedTouches[0].clientX;
  }, { passive: true });
  stage.addEventListener('touchend', event => {
    if (touchStart !== null) {
      const delta = event.changedTouches[0].clientX - touchStart;
      if (Math.abs(delta) > 45) show(index + (delta < 0 ? 1 : -1));
    }
    touchStart = null;
  }, { passive: true });

  fit();
  show(fromHash());
})();
