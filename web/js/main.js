/**
 * WristControl - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
  // Mobile Navigation Toggle
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      navLinks.classList.toggle('active');
      navToggle.setAttribute('aria-expanded',
        navLinks.classList.contains('active'));
    });

    // Close mobile nav when clicking a link
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('active');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href === '#') return;

      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        const headerOffset = 80;
        const elementPosition = target.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    });
  });

  // Header background on scroll
  const header = document.querySelector('.header');
  if (header) {
    const updateHeaderBackground = () => {
      if (window.scrollY > 50) {
        header.style.background = 'rgba(17, 24, 39, 0.95)';
      } else {
        header.style.background = 'rgba(17, 24, 39, 0.8)';
      }
    };

    window.addEventListener('scroll', updateHeaderBackground, { passive: true });
    updateHeaderBackground();
  }

  // Animate elements on scroll
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-fade-in-up');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Observe feature cards and gesture items
  document.querySelectorAll('.feature-card, .gesture-item, .stat-item').forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
  });

  // Platform detection for download button
  const detectPlatform = () => {
    const platform = navigator.platform.toLowerCase();
    if (platform.includes('win')) return 'windows';
    if (platform.includes('mac')) return 'macos';
    if (platform.includes('linux')) return 'linux';
    return 'unknown';
  };

  // Update download button text based on platform
  const downloadBtn = document.querySelector('.cta-section .btn-primary');
  if (downloadBtn) {
    const platform = detectPlatform();
    const platformNames = {
      windows: 'Windows',
      macos: 'macOS',
      linux: 'Linux'
    };
    if (platformNames[platform]) {
      downloadBtn.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download for ${platformNames[platform]}
      `;
    }
  }

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    // ESC to close mobile nav
    if (e.key === 'Escape' && navLinks?.classList.contains('active')) {
      navLinks.classList.remove('active');
      navToggle?.setAttribute('aria-expanded', 'false');
    }
  });

  console.log('WristControl website initialized');
});
