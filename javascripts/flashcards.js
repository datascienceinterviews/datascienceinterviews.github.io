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

    // Create backdrop for mobile
    const backdrop = document.createElement('div');
    backdrop.className = 'dropdown-backdrop';
    document.body.appendChild(backdrop);



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
            padding: 0;
            border-radius: 8px;
            z-index: 10;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            max-height: 400px;
            overflow-y: auto;
            min-width: 280px;
            margin-top: 0.5rem;
        }
        .topics-dropdown.hidden { display: none; }

        .topics-dropdown-header {
            position: sticky;
            top: 0;
            background: var(--md-primary-fg-color, #7c4dff);
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            font-size: 0.9rem;
            border-bottom: 2px solid rgba(255,255,255,0.2);
            z-index: 1;
        }

        .dropdown-close-btn {
            display: none; /* Hidden on desktop */
        }

        .topics-dropdown-content {
            padding: 0.75rem;
        }

        .topic-checkbox {
            display: flex;
            align-items: center;
            padding: 0.6rem 0.75rem;
            margin-bottom: 0.35rem;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }

        .topic-checkbox:hover {
            background: var(--md-primary-fg-color--light, rgba(124, 77, 255, 0.08));
            border-color: var(--md-primary-fg-color--light, rgba(124, 77, 255, 0.2));
        }

        .topic-checkbox input[type="checkbox"] {
            margin: 0;
            margin-right: 0.65rem;
            cursor: pointer;
            width: 18px;
            height: 18px;
            accent-color: var(--md-primary-fg-color, #7c4dff);
        }

        .topic-checkbox-label {
            flex: 1;
            font-size: 0.9rem;
            user-select: none;
        }

        .topic-count {
            font-size: 0.8rem;
            color: var(--md-default-fg-color--light);
            background: var(--md-code-bg-color, #f5f5f5);
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 600;
            margin-left: 0.5rem;
        }

        .select-all-link {
            display: block;
            text-align: center;
            padding: 0.6rem;
            margin: 0.5rem;
            background: var(--md-default-fg-color--lightest, #f5f5f5);
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--md-primary-fg-color, #7c4dff);
            text-decoration: none;
            transition: all 0.15s ease-in-out;
            border: 1px solid transparent;
        }

        .select-all-link:hover {
            background: var(--md-primary-fg-color, #7c4dff);
            color: white;
            border-color: var(--md-primary-fg-color, #7c4dff);
            box-shadow: 0 2px 8px rgba(124, 77, 255, 0.3);
        }

        .select-all-link:active {
            transform: scale(0.98);
        }

        [data-md-color-scheme="slate"] .topics-dropdown {
            background: var(--md-default-bg-color);
            border-color: var(--md-default-fg-color--lightest);
        }

        [data-md-color-scheme="slate"] .topic-count {
            background: rgba(255, 255, 255, 0.1);
            color: var(--md-default-fg-color--light);
        }

        /* Mobile Responsive Styles */
        @media (max-width: 768px) {
            .topics-dropdown {
                position: fixed;
                top: auto;
                bottom: 0;
                left: 0;
                right: 0;
                max-height: 70vh;
                min-width: 100%;
                width: 100%;
                border-radius: 16px 16px 0 0;
                margin-top: 0;
                animation: slideUp 0.3s ease-out;
                background: var(--md-default-bg-color, white) !important;
            }

            @keyframes slideUp {
                from {
                    transform: translateY(100%);
                }
                to {
                    transform: translateY(0);
                }
            }

            .topics-dropdown-header {
                border-radius: 16px 16px 0 0;
                padding: 1rem;
                font-size: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .dropdown-close-btn {
                display: flex !important; /* Show on mobile */
                background: rgba(255, 255, 255, 0.2);
                border: none;
                color: white;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                font-size: 1.2rem;
                cursor: pointer;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                padding: 0;
                line-height: 1;
            }

            .dropdown-close-btn:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: scale(1.1);
            }

            .topics-dropdown-content {
                padding: 1rem;
                background: var(--md-default-bg-color, white);
            }

            .topic-checkbox {
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }

            .topic-checkbox input[type="checkbox"] {
                width: 20px;
                height: 20px;
            }

            .topic-checkbox-label {
                font-size: 1rem;
            }

            .topic-count {
                font-size: 0.85rem;
                padding: 4px 10px;
            }

            .select-all-link {
                padding: 0.75rem;
                font-size: 0.9rem;
            }
        }

        /* Backdrop overlay for mobile */
        .dropdown-backdrop {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 9;
        }

        @media (max-width: 768px) {
            .dropdown-backdrop.active {
                display: block;
            }
        }

        @media (max-width: 480px) {
            .flashcard-container {
                padding: 0 0.5rem;
            }

            .action-buttons-row {
                flex-direction: column;
                gap: 0.75rem !important;
            }

            .action-buttons-row > * {
                width: 100%;
            }

            #toggle-topics-btn, #shuffle-btn {
                width: 100%;
            }
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

    // Helper function to format topic names with proper capitalization
    function formatTopicName(topic) {
        // Remove redundant text
        let cleaned = topic
            .replace(/interview questions?/gi, '')
            .replace(/\s+/g, ' ')
            .trim();

        // Special cases for acronyms and specific terms
        const specialCases = {
            'nlp': 'NLP',
            'natural language processing': 'NLP',
            'sql': 'SQL',
            'ab testing': 'A/B Testing',
            'a/b testing': 'A/B Testing',
            'dsa': 'DSA',
            'data structures & algorithms': 'DSA',
            'data structures and algorithms': 'DSA',
            'numpy': 'NumPy',
            'pandas': 'Pandas',
            'scikit learn': 'Scikit-Learn',
            'scikit-learn': 'Scikit-Learn',
            'langchain': 'LangChain',
            'langgraph': 'LangGraph',
            'sklearn': 'Scikit-Learn',
            'machine learning': 'ML',
            'system design': 'System Design',
            'probability': 'Probability'
        };

        const lowerCleaned = cleaned.toLowerCase();
        if (specialCases[lowerCleaned]) {
            return specialCases[lowerCleaned];
        }

        // Title case for other topics
        return cleaned.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

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

            // Extract Topics and sort by popularity (question count)
            const topicCounts = {};
            allQuestions.forEach(q => {
                topicCounts[q.topic] = (topicCounts[q.topic] || 0) + 1;
            });
            const topics = [...new Set(allQuestions.map(q => q.topic))].sort((a, b) => {
                // Sort by count (descending), then alphabetically
                const countDiff = topicCounts[b] - topicCounts[a];
                return countDiff !== 0 ? countDiff : a.localeCompare(b);
            });

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
                const isHidden = topicsDropdown.classList.toggle('hidden');
                if (!isHidden) {
                    backdrop.classList.add('active');
                } else {
                    backdrop.classList.remove('active');
                }
            });
            document.getElementById('shuffle-btn').addEventListener('click', () => {
                shuffleArray(filteredQuestions);
                currentFilteredIndex = 0;
                isFlipped = false;
                updateCardFlip();
                renderCard();
            });

            // Close dropdown if clicking outside or on backdrop
            document.addEventListener('click', (e) => {
                if (!topicsDropdown.contains(e.target) && e.target.id !== 'toggle-topics-btn') {
                    topicsDropdown.classList.add('hidden');
                    backdrop.classList.remove('active');
                }
            });

            // Close dropdown when clicking backdrop
            backdrop.addEventListener('click', () => {
                topicsDropdown.classList.add('hidden');
                backdrop.classList.remove('active');
            });

        } catch (e) {
            console.error(e);
            app.innerHTML = `<div class="admonition failure"><p class="admonition-title">Error</p><p>Could not load flashcards. Please try again later. ${e.message}</p></div>`;
        }
    }

    function renderTopicsDropdown(topics) {
        topicsDropdown.innerHTML = '';

        // Calculate question counts per topic
        const topicCounts = {};
        allQuestions.forEach(q => {
            topicCounts[q.topic] = (topicCounts[q.topic] || 0) + 1;
        });

        // Calculate total selected questions
        const selectedCount = topics.filter(t => selectedTopics[t])
            .reduce((sum, t) => sum + (topicCounts[t] || 0), 0);
        const totalCount = allQuestions.length;

        // Header
        const header = document.createElement('div');
        header.className = 'topics-dropdown-header';

        const headerText = document.createElement('span');
        headerText.textContent = `Filter Topics (${selectedCount}/${totalCount} questions)`;
        header.appendChild(headerText);

        // Close button (visible on mobile)
        const closeBtn = document.createElement('button');
        closeBtn.className = 'dropdown-close-btn';
        closeBtn.innerHTML = '×';
        closeBtn.onclick = (e) => {
            e.stopPropagation();
            topicsDropdown.classList.add('hidden');
            backdrop.classList.remove('active');
        };
        header.appendChild(closeBtn);

        topicsDropdown.appendChild(header);

        // Content wrapper
        const content = document.createElement('div');
        content.className = 'topics-dropdown-content';

        // Select All / None link
        const selectAllLink = document.createElement('a');
        selectAllLink.href = "#";
        selectAllLink.className = 'select-all-link';
        selectAllLink.textContent = "Select All / None";
        selectAllLink.onclick = (e) => {
            e.preventDefault();
            const allSelected = topics.every(t => selectedTopics[t]);
            topics.forEach(t => selectedTopics[t] = !allSelected);
            renderTopicsDropdown(topics); // Re-render checkboxes
            applyFilter();
        };
        content.appendChild(selectAllLink);

        // Topic checkboxes
        topics.forEach(t => {
            const label = document.createElement('label');
            label.className = 'topic-checkbox';

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = selectedTopics[t];
            cb.onchange = (e) => {
                selectedTopics[t] = e.target.checked;
                localStorage.setItem('flashcard-topics', JSON.stringify(selectedTopics));
                renderTopicsDropdown(topics); // Update header count
                applyFilter();
            };

            const textSpan = document.createElement('span');
            textSpan.className = 'topic-checkbox-label';
            textSpan.textContent = formatTopicName(t);

            const countSpan = document.createElement('span');
            countSpan.className = 'topic-count';
            countSpan.textContent = topicCounts[t] || 0;

            label.appendChild(cb);
            label.appendChild(textSpan);
            label.appendChild(countSpan);
            content.appendChild(label);
        });

        topicsDropdown.appendChild(content);
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
        document.getElementById('card-topic').textContent = formatTopicName(q.topic);
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
