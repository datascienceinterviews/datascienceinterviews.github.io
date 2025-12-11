document.addEventListener('DOMContentLoaded', function () {
    const app = document.getElementById('flashcard-app');
    if (!app) return;

    // --- State ---
    let allQuestions = [];
    let filteredQuestions = [];
    let currentFilteredIndex = 0;

    // Topics: { "Python": true, "SQL": false, ... }
    let selectedTopics = {};
    let isFlipped = false;

    // --- DOM Elements ---
    app.innerHTML = `
        <div class="flashcard-container">
            
            <!-- Buttons Row (Moved to Top) -->
            <div class="action-buttons-row" style="display: flex; justify-content: space-between; gap: 1rem; align-items: center; margin-bottom: 1.5rem;">
                <!-- Filter: Standard Button Size -->
                <div class="controls-area" style="margin: 0;"> 
                    <button id="toggle-topics-btn" class="md-button" style="border: 1px solid #7c4dff; color: #7c4dff;">Filter Topics</button>
                    <div id="topics-dropdown" class="topics-dropdown hidden"></div>
                </div>

                <!-- Shuffle: Standard Button Size -->
                <button id="shuffle-btn" class="md-button md-button--primary" style="margin: 0;">Shuffle 🔀</button>
            </div>

            <!-- Flashcard Itself -->
            <div class="card-area">
                <div class="card-scene">
                    <div class="card" id="flashcard">
                        <div class="card-face card-front">
                            <div class="card-header">
                                <span class="badge topic-badge" id="card-topic">Topic</span>
                                <span class="badge difficulty-badge" id="card-difficulty">Difficulty</span>
                            </div>
                            <div class="card-content">
                                <h3 id="card-question">Loading...</h3>
                                <div class="companies-asked">
                                    <strong>Asked by:</strong> <span id="card-companies"></span>
                                </div>
                            </div>
                            <div class="card-footer">
                                <span class="click-hint">Click to flip</span>
                            </div>
                        </div>
                        <div class="card-face card-back">
                             <div class="card-header">
                                <span class="badge">Answer</span>
                            </div>
                            <div class="card-content markdown-body" id="card-answer">
                                <!-- Answer content -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Navigation -->
            <div class="navigation-controls">
                <button id="prev-btn" class="md-button">Previous</button>
                <span id="progress-indicator">0 / 0</span>
                <button id="next-btn" class="md-button md-button--primary">Next</button>
            </div>
            
            <!-- Bottom Controls -->
            <div class="controls-area-container" style="margin-top: 2rem; border-top: 1px solid var(--md-default-fg-color--lightest); padding-top: 1.5rem;">
                <!-- Intro Text (BELOW) -->
                <p style="text-align: center; color: var(--md-default-fg-color--light); font-size: 0.9rem;">
                    Test your knowledge with these interactive flashcards! Select topics to customize your session.
                </p>
            </div>
        </div>

    `;

    const card = document.getElementById('flashcard');
    const topicsDropdown = document.getElementById('topics-dropdown');



    // --- Styles Injection (Programmatic to keep it self-contained or use extra.css) ---
    const style = document.createElement('style');
    style.textContent = `
        .flashcard-container {
            max-width: 800px;
            margin: 0 auto;
            font-family: var(--md-text-font-family, sans-serif);
        }
        .controls-area {
            position: relative;
            margin-bottom: 1rem;
            display: flex;
            justify-content: flex-end;
        }
        .topics-dropdown {
            position: absolute;
            top: 100%;
            right: 0;
            background: var(--md-default-bg-color, white);
            border: 1px solid var(--md-default-fg-color--lightest, #ddd);
            padding: 1rem;
            border-radius: 4px;
            z-index: 10;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            max-height: 300px;
            overflow-y: auto;
            min-width: 200px;
        }
        .topics-dropdown.hidden { display: none; }
        .topic-checkbox {
            display: block;
            margin-bottom: 0.5rem;
            cursor: pointer;
        }
        
        /* Card Flip Animation */
        .card-scene {
            perspective: 1000px;
            height: 400px; /* Fixed height for consistency */
            margin-bottom: 1.5rem;
        }
        .card {
            width: 100%;
            height: 100%;
            position: relative;
            transition: transform 0.6s;
            transform-style: preserve-3d;
            cursor: pointer;
        }
        .card.is-flipped {
            transform: rotateY(180deg);
        }
        .card-face {
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            border-radius: 12px;
            padding: 2rem;
            background: var(--md-default-bg-color, white);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            border: 1px solid var(--md-default-fg-color--lightest, #eee);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        
        .card-face.card-back {
            transform: rotateY(180deg);
            background: var(--md-code-bg-color, #f5f5f5);
        }
        /* Dark mode adjustment if relevant variables exist, otherwise rely on md variables */
        [data-md-color-scheme="slate"] .card-face {
           background: var(--md-default-bg-color);
           border-color: var(--md-default-fg-color--lightest);
        }
        [data-md-color-scheme="slate"] .card-face.card-back {
            background: #2e303e; /* slightly different than bg */
        }

        /* Glow Effects - Prominent */
        .card.glow-easy .card-face {
            box-shadow: 0 0 30px rgba(76, 175, 80, 0.6);
            border: 2px solid #4caf50;
        }
        .card.glow-medium .card-face {
            box-shadow: 0 0 30px rgba(255, 183, 77, 0.6);
            border: 2px solid #ffb74d;
        }
        .card.glow-hard .card-face {
            box-shadow: 0 0 30px rgba(233, 30, 99, 0.6);
            border: 2px solid #e91e63;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
            font-size: 0.85em;
        }
        .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.75rem;
        }
        .topic-badge { background: var(--md-primary-fg-color, #5e35b1); color: white; }
        .difficulty-badge { border: 1px solid currentColor; }
        
        .card-content {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        #card-answer {
            align-items: flex-start;
            text-align: left;
            display: block; /* markdown content */
        }
        .card-footer {
            margin-top: auto;
            text-align: center;
            font-size: 0.8rem;
            color: var(--md-default-fg-color--light);
        }
        
        .navigation-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }
        .companies-asked {
            margin-top: 1rem;
            font-size: 0.9rem;
            color: var(--md-default-fg-color--light);
        }

        #progress-indicator {
            background: var(--md-default-bg-color); /* Match card theme */
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-family: monospace; /* Tech feel */
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border: 1px solid var(--md-default-fg-color--lightest);
            color: var(--md-primary-fg-color);
            min-width: 80px;
            text-align: center;
        }
    `;
    document.head.appendChild(style);

    // --- Logic ---

    async function init() {
        try {
            // Try connection to root assets (assuming page is at /flashcards/)
            // We use ../assets/questions.json because we are at /flashcards/index.html
            const resp = await fetch('../assets/questions.json');
            if (!resp.ok) {
                // Fallback attempt for different URL structures or if hosted at root without pretty urls
                const resp2 = await fetch('assets/questions.json');
                if (resp2.ok) {
                    allQuestions = await resp2.json();
                } else {
                    throw new Error(`Failed to load questions.json (Status: ${resp.status})`);
                }
            } else {
                allQuestions = await resp.json();
            }

            // Extract Topics
            const topics = [...new Set(allQuestions.map(q => q.topic))].sort();

            // Load saved preferences or default all to true
            const savedTopics = JSON.parse(localStorage.getItem('flashcard-topics') || '{}');

            topics.forEach(t => {
                selectedTopics[t] = savedTopics.hasOwnProperty(t) ? savedTopics[t] : true;
            });

            renderTopicsDropdown(topics);
            applyFilter();

            // Event Listeners
            card.addEventListener('click', () => {
                isFlipped = !isFlipped;
                updateCardFlip();
            });

            document.getElementById('next-btn').addEventListener('click', () => nextCard());
            document.getElementById('prev-btn').addEventListener('click', () => prevCard());
            document.getElementById('toggle-topics-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                topicsDropdown.classList.toggle('hidden');
            });
            document.getElementById('shuffle-btn').addEventListener('click', () => {
                shuffleArray(filteredQuestions);
                currentFilteredIndex = 0;
                isFlipped = false;
                updateCardFlip();
                renderCard();
            });

            // Close dropdown if clicking outside
            document.addEventListener('click', (e) => {
                if (!topicsDropdown.contains(e.target) && e.target.id !== 'toggle-topics-btn') {
                    topicsDropdown.classList.add('hidden');
                }
            });

        } catch (e) {
            console.error(e);
            app.innerHTML = `<div class="admonition failure"><p class="admonition-title">Error</p><p>Could not load flashcards. Please try again later. ${e.message}</p></div>`;
        }
    }

    function renderTopicsDropdown(topics) {
        topicsDropdown.innerHTML = '';
        const selectAllLink = document.createElement('a');
        selectAllLink.href = "#";
        selectAllLink.textContent = "Select All / None";
        selectAllLink.style.display = "block";
        selectAllLink.style.marginBottom = "0.5rem";
        selectAllLink.style.fontSize = "0.8rem";
        selectAllLink.onclick = (e) => {
            e.preventDefault();
            const allSelected = topics.every(t => selectedTopics[t]);
            topics.forEach(t => selectedTopics[t] = !allSelected);
            renderTopicsDropdown(topics); // Re-render checkboxes
            applyFilter();
        };
        topicsDropdown.appendChild(selectAllLink);

        topics.forEach(t => {
            const label = document.createElement('label');
            label.className = 'topic-checkbox';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = selectedTopics[t];
            cb.onchange = (e) => {
                selectedTopics[t] = e.target.checked;
                localStorage.setItem('flashcard-topics', JSON.stringify(selectedTopics));
                applyFilter();
            };
            label.appendChild(cb);
            label.appendChild(document.createTextNode(` ${t}`));
            topicsDropdown.appendChild(label);
        });
    }

    function applyFilter() {
        filteredQuestions = allQuestions.filter(q => selectedTopics[q.topic]);
        if (filteredQuestions.length === 0) {
            // Handle empty state
        }
        currentFilteredIndex = 0;
        isFlipped = false;
        updateCardFlip();
        renderCard();
    }

    function renderCard() {
        const q = filteredQuestions[currentFilteredIndex];
        const progressEl = document.getElementById('progress-indicator');
        const count = filteredQuestions.length;

        progressEl.textContent = count > 0 ? `${currentFilteredIndex + 1} / ${count}` : "0 / 0";

        if (!q) {
            document.getElementById('card-question').textContent = "No questions match your filter.";
            document.getElementById('card-companies').textContent = "";
            document.getElementById('card-answer').innerHTML = "";
            return;
        }

        // Front
        document.getElementById('card-topic').textContent = q.topic;
        document.getElementById('card-difficulty').textContent = q.difficulty.replace(/\*\*|🔴|🟡|🟢/g, '').trim(); // Strip formatting if raw markdown leaked

        // Remove existing glow classes
        const cardContainer = document.getElementById('flashcard');
        cardContainer.classList.remove('glow-easy', 'glow-medium', 'glow-hard');

        let diffColor = '#4caf50'; // Default Green
        if (q.difficulty.includes('Hard')) {
            diffColor = '#e91e63';
            cardContainer.classList.add('glow-hard');
        } else if (q.difficulty.includes('Medium')) {
            diffColor = '#ffb74d';
            cardContainer.classList.add('glow-medium');
        } else {
            cardContainer.classList.add('glow-easy');
        }

        document.getElementById('card-difficulty').style.color = diffColor;
        document.getElementById('card-difficulty').style.borderColor = diffColor;

        document.getElementById('card-question').textContent = q.question.replace(/^\d+\.\s*/, ''); // Remove leading number if present
        document.getElementById('card-companies').textContent = q.companies || "Various";

        // Back
        // Content is now pre-rendered HTML from the hook
        const answerEl = document.getElementById('card-answer');
        answerEl.innerHTML = q.answer;

        // Trigger MathJax if available
        if (window.MathJax) {
            if (window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise([answerEl]);
            } else if (window.MathJax.Hub) {
                window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, answerEl]);
            }
        }
    }

    // Minimal Markdown Parser removed as we now use server-side rendering
    function renderMarkdown(text) {
        return text;
    }

    function updateCardFlip() {
        if (isFlipped) card.classList.add('is-flipped');
        else card.classList.remove('is-flipped');
    }

    function nextCard() {
        if (currentFilteredIndex < filteredQuestions.length - 1) {
            currentFilteredIndex++;
            isFlipped = false;
            updateCardFlip();
            setTimeout(renderCard, 300); // Wait for flip back if it was flipped? No, instant snap back is better UI often, or flip then change.
            // Better UX: Flip back immediately then content change.
            renderCard();
        }
    }

    function prevCard() {
        if (currentFilteredIndex > 0) {
            currentFilteredIndex--;
            isFlipped = false;
            updateCardFlip();
            renderCard();
        }
    }

    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    init();
});
