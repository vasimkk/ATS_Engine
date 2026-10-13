document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("resume-upload");
    const jdText = document.getElementById("jd-text");
    const analyzeBtn = document.getElementById("analyze-btn");
    
    let uploadedFile = null;

    // Drag and Drop Logic
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (file.type === "application/pdf") {
            uploadedFile = file;
            dropZone.innerHTML = `<span class="icon">✅</span><p>${file.name}</p>`;
            checkFormReady();
        } else {
            alert("Please upload a PDF file.");
        }
    }

    jdText.addEventListener("input", checkFormReady);

    function checkFormReady() {
        if (uploadedFile && jdText.value.trim().length > 10) {
            analyzeBtn.disabled = false;
        } else {
            analyzeBtn.disabled = true;
        }
    }

    // Animation and API Call
    analyzeBtn.addEventListener("click", async () => {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "Analyzing Match...";

        try {
            // 1. Upload Resume
            const formData = new FormData();
            formData.append("file", uploadedFile);
            
            const resumeRes = await fetch("/api/v1/resumes/upload", {
                method: "POST",
                body: formData
            });
            const resumeData = await resumeRes.json();
            const resumeId = resumeData.resume_id;

            // 2. Upload Job Description
            await fetch("/api/v1/jobs/upload", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    resume_id: resumeId,
                    job_descriptions: [jdText.value]
                })
            });

            // 3. Get Match Score
            const matchRes = await fetch(`/api/v1/jobs/match/${resumeId}`);
            const matchData = await matchRes.json();
            
            let finalScore = 0;
            let semanticScore = 0;
            let fuzzyScore = 0;
            let formatScore = 0;
            let missingKeywords = [];

            if (matchData.top_matches && matchData.top_matches.length > 0) {
                const bestMatch = matchData.top_matches[0];
                finalScore = Math.round(bestMatch.match_percentage);
                semanticScore = Math.round(bestMatch.semantic_score);
                fuzzyScore = Math.round(bestMatch.fuzzy_score);
                formatScore = Math.round(bestMatch.format_score);
                missingKeywords = bestMatch.missing_keywords || [];
            }

            // Animate Results
            animateScore(finalScore);
            
            document.getElementById("semantic-score").textContent = semanticScore + "%";
            document.getElementById("fuzzy-score").textContent = fuzzyScore + "%";
            document.getElementById("format-score").textContent = formatScore + "%";
            
            // Render Missing Keywords
            const insightsSection = document.getElementById("actionable-insights");
            const tagsContainer = document.getElementById("missing-tags");
            tagsContainer.innerHTML = ""; // clear previous
            
            if (missingKeywords.length > 0) {
                insightsSection.style.display = "block";
                missingKeywords.forEach(keyword => {
                    const tag = document.createElement("span");
                    tag.className = "keyword-tag";
                    tag.textContent = keyword;
                    tagsContainer.appendChild(tag);
                });
            } else {
                insightsSection.style.display = "none";
            }

        } catch (error) {
            console.error("Error analyzing:", error);
            alert("Analysis failed. Please check the backend console.");
            // For demo purposes, run animation anyway
            animateScore(Math.floor(Math.random() * 40) + 50);
        }

        analyzeBtn.textContent = "Analyze Match";
        analyzeBtn.disabled = false;
    });

    function animateScore(targetScore) {
        const circle = document.querySelector('.progress-ring__circle.fg');
        const radius = circle.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        
        circle.style.strokeDasharray = `${circumference} ${circumference}`;
        circle.style.strokeDashoffset = circumference;
        
        const offset = circumference - (targetScore / 100) * circumference;
        
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 100);

        const scoreElement = document.getElementById("final-score");
        let currentScore = 0;
        const interval = setInterval(() => {
            if (currentScore >= targetScore) {
                clearInterval(interval);
                scoreElement.textContent = targetScore;
            } else {
                currentScore++;
                scoreElement.textContent = currentScore;
            }
        }, 15);
    }
});
