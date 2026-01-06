/**
 * WristControl - Documentation JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
  // Copy code functionality
  document.querySelectorAll('.docs-copy-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const targetId = btn.dataset.copy;
      const codeElement = document.getElementById(targetId);

      if (codeElement) {
        try {
          await navigator.clipboard.writeText(codeElement.textContent);
          const originalText = btn.textContent;
          btn.textContent = 'Copied!';
          btn.style.background = 'var(--color-success)';

          setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
          }, 2000);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      }
    });
  });

  // Active navigation highlighting
  const navLinks = document.querySelectorAll('.docs-nav-section a');
  const sections = document.querySelectorAll('.docs-section');

  const observerOptions = {
    rootMargin: '-100px 0px -70% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinks.forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => {
    if (section.id) {
      observer.observe(section);
    }
  });

  // Smooth scroll for sidebar links
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href.startsWith('#')) {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
          history.pushState(null, null, href);
        }
      }
    });
  });

  // Handle initial hash
  if (window.location.hash) {
    const target = document.querySelector(window.location.hash);
    if (target) {
      setTimeout(() => {
        target.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }

  console.log('Documentation page initialized');
});
