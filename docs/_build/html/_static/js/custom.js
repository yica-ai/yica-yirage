// YICA/YiRage Custom JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Add copy functionality to code blocks
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach(function(block) {
        if (!block.querySelector('.copybutton')) {
            const button = document.createElement('button');
            button.className = 'copybutton';
            button.innerHTML = '📋';
            button.title = 'Copy to clipboard';
            button.onclick = function() {
                navigator.clipboard.writeText(block.textContent).then(function() {
                    button.innerHTML = '✅';
                    setTimeout(function() {
                        button.innerHTML = '📋';
                    }, 2000);
                });
            };
            block.style.position = 'relative';
            block.appendChild(button);
        }
    });

    // Add smooth scrolling to anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Add external link indicators
    const externalLinks = document.querySelectorAll('a[href^="http"]');
    externalLinks.forEach(function(link) {
        if (!link.hostname.includes(window.location.hostname)) {
            link.innerHTML += ' 🔗';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
        }
    });

    // Performance metrics display
    if (typeof performance !== 'undefined' && performance.timing) {
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        console.log('YICA/YiRage Documentation loaded in ' + loadTime + 'ms');
    }
});
