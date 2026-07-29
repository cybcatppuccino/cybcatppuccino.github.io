(() => {
  const detailsBoxes = [...document.querySelectorAll('details.box')];
  const expandButton = document.getElementById('expandAll');
  const collapseButton = document.getElementById('collapseAll');

  function setAllDetails(open) {
    if (!detailsBoxes.length) return;
    expandButton.disabled = true;
    collapseButton.disabled = true;
    const source = open ? detailsBoxes.filter(x => !x.open) : detailsBoxes.filter(x => x.open);
    let index = 0;
    const batchSize = 5;
    function step() {
      const end = Math.min(index + batchSize, source.length);
      for (; index < end; index += 1) source[index].open = open;
      if (index < source.length) {
        requestAnimationFrame(step);
      } else {
        expandButton.disabled = false;
        collapseButton.disabled = false;
        scheduleProgress();
      }
    }
    requestAnimationFrame(step);
  }

  expandButton?.addEventListener('click', () => setAllDetails(true));
  collapseButton?.addEventListener('click', () => setAllDetails(false));
  document.getElementById('printPage')?.addEventListener('click', () => window.print());

  const glossary = document.getElementById('glossary');
  document.querySelector('a[href="#glossary"]')?.addEventListener('click', () => {
    if (glossary) glossary.open = true;
  });

  const progress = document.getElementById('progress');
  let progressFrame = 0;
  function updateProgress() {
    progressFrame = 0;
    const max = document.documentElement.scrollHeight - innerHeight;
    const ratio = max > 0 ? Math.min(1, Math.max(0, scrollY / max)) : 0;
    if (progress) progress.style.transform = `scaleX(${ratio})`;
  }
  function scheduleProgress() {
    if (!progressFrame) progressFrame = requestAnimationFrame(updateProgress);
  }
  addEventListener('scroll', scheduleProgress, {passive: true});
  addEventListener('resize', scheduleProgress, {passive: true});
  document.addEventListener('toggle', scheduleProgress, true);
  scheduleProgress();

  const links = new Map(
    [...document.querySelectorAll('.toc a[href^="#lecture-"]')]
      .map(a => [a.getAttribute('href').slice(1), a])
  );
  let activeLink = null;
  const observer = new IntersectionObserver(entries => {
    const visible = entries
      .filter(entry => entry.isIntersecting)
      .sort((a, b) => Math.abs(a.boundingClientRect.top) - Math.abs(b.boundingClientRect.top))[0];
    if (!visible) return;
    const next = links.get(visible.target.id);
    if (next === activeLink) return;
    activeLink?.classList.remove('active');
    next?.classList.add('active');
    activeLink = next || null;
  }, {rootMargin: '-12% 0px -78% 0px'});
  document.querySelectorAll('.lecture-card').forEach(card => observer.observe(card));

  let printState = [];
  addEventListener('beforeprint', () => {
    printState = detailsBoxes.map(x => x.open);
    detailsBoxes.forEach(x => { x.open = true; });
    if (glossary) glossary.open = true;
  });
  addEventListener('afterprint', () => {
    detailsBoxes.forEach((x, i) => { x.open = printState[i]; });
    scheduleProgress();
  });
})();
