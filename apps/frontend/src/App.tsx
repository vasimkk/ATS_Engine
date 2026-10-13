import React, { useState, useEffect } from 'react';

function App() {
  const [isLoaded, setIsLoaded] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [matchData, setMatchData] = useState<any>(null);
  const [selectedMatch, setSelectedMatch] = useState<any>(null);
  const [aiReviewLoading, setAiReviewLoading] = useState(false);
  const [aiReviewData, setAiReviewData] = useState<string | null>(null);
  const [gapAnalysisLoading, setGapAnalysisLoading] = useState(false);
  const [gapAnalysisData, setGapAnalysisData] = useState<any>(null);
  const [skip, setSkip] = useState(0);

  const handleGapAnalysis = async () => {
    if (!selectedMatch) return;
    setGapAnalysisLoading(true);
    setGapAnalysisData(null);
    const payload = {
      job_id: selectedMatch.job_id,
      resume_id: matchData?.resume_id || "temp-resume-123",
      resume_text: matchData?.parsed_text_preview || ""
    };

    try {
      const res = await fetch('http://localhost:8000/api/v1/resumes/gap-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === 'success') {
        setGapAnalysisData(data.data);
      } else {
        alert(data.message || 'Failed to generate Gap Analysis');
      }
    } catch (err) {
      console.error(err);
      alert('Network error communicating with AI');
    } finally {
      setGapAnalysisLoading(false);
    }
  };

  const handleAskAI = async () => {
    if (!selectedMatch) return;
    setAiReviewLoading(true);
    setAiReviewData(null);
    const payload = {
      JDid: selectedMatch.job_id,
      Candidateid: "cand_123",
      ResumeID: matchData?.resume_id || "temp-resume-123"
    };

    try {
      const res = await fetch('http://localhost:8000/api/v1/Get_EnhancedResume_for_PoorJDScore', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === 'success') {
        setAiReviewData(`Enhanced ATS Score: ${data.EnhATS.toFixed(1)}\n\n${data.EnhResume}`);
      } else {
        alert(data.message || 'Failed to generate AI review');
      }
    } catch (err) {
      console.error(err);
      alert('Network error communicating with AI');
    } finally {
      setAiReviewLoading(false);
    }
  };

  const [limit, setLimit] = useState(100);
  const [topK, setTopK] = useState(1000);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setIsLoaded(false);
    setSelectedMatch(null);

    const formData = new FormData();
    formData.append('resume_file', file);

    try {
      const res = await fetch(`http://localhost:8000/api/v1/Get_N_JDs_for_Res?N=${limit}&top_k=${topK}`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.status === 'error') {
        throw new Error(data.message || 'Backend returned an error');
      }
      setMatchData(data);
      setTimeout(() => setIsLoaded(true), 100);
    } catch (err: any) {
      console.error(err);
      alert("Failed to calculate ATS score: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadMore = async () => {
    if (!file) return;

    setLoading(true);
    const newSkip = skip + 10;
    setSkip(newSkip);

    const lastJDid = matchData?.matches?.[matchData.matches.length - 1]?.job_id || "";
    const payload = {
      N: limit,
      LastJDid: lastJDid,
      ResumeID: matchData?.resume_id || "temp-resume-123",
      top_k: topK
    };

    try {
      const res = await fetch(`http://localhost:8000/api/v1/Get_N_moreJDs_for_Res`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === 'error') {
        throw new Error(data.message || 'Backend returned an error');
      }
      setMatchData(prev => ({
        ...prev,
        matches: [...(prev?.matches || []), ...(data.matches || [])]
      }));
    } catch (err: any) {
      console.error(err);
      alert("Failed to load more jobs: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const extractedSkills = matchData?.extracted_skills || [];
  const previewText = matchData?.parsed_text_preview || "Upload a PDF to see extraction...";
  const matches = [...(matchData?.matches || [])].sort((a: any, b: any) => (b.overall_score || 0) - (a.overall_score || 0));

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-emerald-400';
    if (score >= 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  // Helper variables for the Detailed View
  let overallScore = 0, domainScore = 0, skScore = 0, expFitScore = 0, formatScore = 0;
  let missingSkills: string[] = [];
  let missingRequiredSkills: string[] = [];
  let missingPreferredSkills: string[] = [];
  let debugMath: any = {};

  if (selectedMatch) {
    overallScore = selectedMatch.overall_score || 0;
    domainScore = selectedMatch.score_data?.sub_scores?.domain_score || 0;
    skScore = selectedMatch.score_data?.sub_scores?.skills_coverage || 0;
    expFitScore = selectedMatch.score_data?.sub_scores?.experience_fit || 0;
    formatScore = selectedMatch.score_data?.sub_scores?.section_completeness || 0;
    missingSkills = selectedMatch.score_data?.missing_keywords || [];
    missingRequiredSkills = selectedMatch.score_data?.missing_required_skills || [];
    missingPreferredSkills = selectedMatch.score_data?.missing_preferred_skills || [];
    debugMath = selectedMatch.score_data?.debug_math || {};
  }

  const {
    resume_yoe = 0,
    jd_yoe = 0,
    yoe_penalty = 0,
    found_sections = 0,
    skill_w = 0.6,
    exp_w = 0.4,
  } = debugMath;

  const llmScoreMatch = aiReviewData ? aiReviewData.match(/\*\*Final LLM Score\*\*:\s*(\d+)/i) : null;
  const parsedLlmScore = llmScoreMatch ? parseInt(llmScoreMatch[1], 10) : null;

  const [simulatedProgress, setSimulatedProgress] = useState(0);

  useEffect(() => {
    if (loading) {
      setSimulatedProgress(0);
      const interval = setInterval(() => {
        setSimulatedProgress(prev => {
          if (prev >= 95) return prev;
          return prev + Math.random() * 5;
        });
      }, 500);
      return () => clearInterval(interval);
    } else {
      setSimulatedProgress(100);
    }
  }, [loading]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-sans selection:bg-blue-600 selection:text-white">
      <div className="max-w-6xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex justify-between items-center border-b border-gray-800 pb-6">
          <div>
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
              ATS Engine
            </h1>
            <p className="text-gray-400 mt-2 text-sm font-mono">
              Direct PDF
            </p>
          </div>
        </div>

        {/* Upload Form - Hidden if looking at a detail view */}
        {!selectedMatch && (
          <form onSubmit={handleSubmit} className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 shadow-xl space-y-6">
            <div>
              <label className="block text-sm font-bold text-gray-400 mb-2">Resume File (PDF, DOCX, TXT, MD, CSV, etc.)</label>
              <input
                type="file"
                accept="*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="w-full bg-black/50 border border-gray-700 rounded p-3 text-sm focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-bold text-gray-400 mb-2">Top N jd's Please enter the no of jd's you want:</label>
                <input
                  type="number"
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
                  className="w-full bg-black/50 border border-gray-700 rounded p-3 text-sm focus:border-blue-500 focus:outline-none"
                  min="1"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={loading || !file}
              className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 font-bold rounded transition-colors"
            >
              {loading ? 'Initializing Batch Processing...' : 'Match Against All Jobs'}
            </button>
          </form>
        )}

        {/* --- LIVE BATCH PROCESSING UI --- */}
        {loading && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 shadow-2xl space-y-8 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gray-800">
              <div
                className="h-full bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.6)] transition-all duration-300 ease-out"
                style={{ width: `${simulatedProgress}%` }}
              ></div>
            </div>

            <div className="flex justify-between items-end">
              <div>
                <h2 className="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500 animate-pulse">
                  Batch Processing Jobs
                </h2>
                <p className="text-sm text-gray-400 mt-1 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
                  GPT-4o-mini is evaluating 10 matches concurrently
                </p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-mono text-blue-400">{Math.min(100, Math.floor(simulatedProgress))}%</div>
                <div className="text-xs text-gray-500 uppercase tracking-widest">Complete</div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[...Array(10)].map((_, i) => {
                const isActive = simulatedProgress > (i * 9);
                const isDone = simulatedProgress > ((i + 1) * 9.5);
                return (
                  <div key={i} className={`p-3 rounded border transition-all duration-500 flex flex-col items-center justify-center gap-2 ${isDone ? 'bg-emerald-900/20 border-emerald-500/50 text-emerald-400' :
                    isActive ? 'bg-blue-900/20 border-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.3)] text-blue-400 animate-pulse' :
                      'bg-black border-gray-800 text-gray-600'
                    }`}>
                    {isDone ? (
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                    ) : isActive ? (
                      <svg className="w-6 h-6 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                    ) : (
                      <svg className="w-6 h-6 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    )}
                    <span className="text-[10px] font-mono uppercase tracking-wider font-bold">
                      {isDone ? 'Scored' : isActive ? 'LLM Analyzing' : 'Pending'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Grid View */}
        {matchData && !selectedMatch && !loading && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold border-b border-gray-800 pb-2">Top Job Matches</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {matches.map((match: any, idx: number) => (
                <div
                  key={match.job_id}
                  onClick={() => setSelectedMatch(match)}
                  className="bg-gray-900/80 border border-gray-800 rounded-xl p-6 flex flex-col hover:border-blue-500 transition-colors cursor-pointer"
                >
                  <div className="flex justify-between items-start mb-4">
                    <span className="text-xs text-gray-500 font-mono">Rank #{idx + 1}</span>
                    <span className={`text-3xl font-black ${getScoreColor(match.overall_score)}`}>
                      {match.overall_score}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300 mb-6 flex-grow">
                    {match.preview}
                  </div>
                  <div className="grid grid-cols-3 gap-2 mt-auto pt-4 border-t border-gray-800">
                    <div className="text-center">
                      <div className="text-[10px] text-gray-500 uppercase">Domain</div>
                      <div className={`font-bold ${getScoreColor(match.score_data?.sub_scores?.domain_score)}`}>{match.score_data?.sub_scores?.domain_score || 0}</div>
                    </div>
                    <div className="text-center border-l border-r border-gray-800">
                      <div className="text-[10px] text-gray-500 uppercase">Skills</div>
                      <div className={`font-bold ${getScoreColor(match.score_data?.sub_scores?.skills_coverage)}`}>{match.score_data?.sub_scores?.skills_coverage || 0}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-[10px] text-gray-500 uppercase">Exp</div>
                      <div className={`font-bold ${getScoreColor(match.score_data?.sub_scores?.experience_fit)}`}>{match.score_data?.sub_scores?.experience_fit || 0}</div>
                    </div>
                  </div>
                  <div className="mt-4 text-center text-xs text-blue-400 font-bold bg-blue-900/20 py-2 rounded">
                    Click for detailed analysis &rarr;
                  </div>
                </div>
              ))}
            </div>
            {matches.length > 0 && (
              <div className="flex justify-center mt-8 pt-4">
                <button
                  onClick={handleLoadMore}
                  disabled={loading}
                  className={`bg-gray-800 hover:bg-gray-700 text-white font-bold py-3 px-8 rounded-lg transition-colors shadow-lg border border-gray-700 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {loading ? 'Loading...' : 'Load Next 10 Jobs'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Detailed Analytics View */}
        {selectedMatch && (
          <div className="space-y-8">
            <button
              onClick={() => {
                setSelectedMatch(null);
                setAiReviewData(null);
              }}
              className="text-sm font-bold text-gray-400 hover:text-white flex items-center gap-2 mb-4"
            >
              &larr; Back to All Matches
            </button>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Panel 1: Final Score (Top/Left span) */}
              <div className="lg:col-span-3 bg-gray-900/50 border border-gray-800 rounded-xl p-8 shadow-2xl backdrop-blur-sm relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-10">
                  <svg className="w-64 h-64" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 22h20L12 2z" /></svg>
                </div>
                <div className="flex flex-col md:flex-row items-center gap-12 z-10 relative">
                  <div className="relative flex items-center justify-center">
                    {/* SVG Progress Ring */}
                    <svg className="w-48 h-48 transform -rotate-90">
                      <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-gray-800" />
                      <circle
                        cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent"
                        strokeDasharray={88 * 2 * Math.PI}
                        strokeDashoffset={(88 * 2 * Math.PI) * (1 - overallScore / 100)}
                        className={`${getScoreColor(overallScore)} transition-all duration-1000 ease-out`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <div className="absolute flex flex-col items-center">
                      <span className={`text-5xl font-black ${getScoreColor(overallScore)}`}>{overallScore}</span>
                      <span className="text-sm text-gray-400 uppercase tracking-widest mt-1">Overall</span>
                    </div>
                  </div>

                  <div className="flex-1 space-y-4">
                    <h2 className="text-2xl font-bold">The Phase 5 Overall Formula</h2>
                    <div className="bg-black/50 p-4 rounded-lg font-mono text-sm text-blue-300 border border-blue-900/50">
                      <span className="text-gray-500"># Option B: Domain acts as a scaling multiplier</span><br />
                      base = ({skScore} * {skill_w}) + ({expFitScore} * {exp_w})<br />
                      overall = base * ({domainScore} / 100) = {overallScore}
                    </div>
                    <div className="grid grid-cols-3 gap-4 mt-4">
                      <div className="bg-gray-800/50 p-4 rounded border border-gray-700">
                        <div className="text-gray-400 text-xs uppercase mb-1">Domain (30%)</div>
                        <div className={`text-xl font-bold ${getScoreColor(domainScore)}`}>{domainScore}</div>
                      </div>
                      <div className="bg-gray-800/50 p-4 rounded border border-gray-700">
                        <div className="text-gray-400 text-xs uppercase mb-1">Skills (45%)</div>
                        <div className={`text-xl font-bold ${getScoreColor(skScore)}`}>{skScore}</div>
                      </div>
                      <div className="bg-gray-800/50 p-4 rounded border border-gray-700">
                        <div className="text-gray-400 text-xs uppercase mb-1">Experience (25%)</div>
                        <div className={`text-xl font-bold ${getScoreColor(expFitScore)}`}>{expFitScore}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Panel 2: Math Engine */}
              <div className="lg:col-span-2 bg-gray-900/50 border border-gray-800 rounded-xl p-6 shadow-2xl">
                <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                  The Math Engine
                </h3>

                <div className="space-y-8">
                  {/* Domain */}
                  <div>
                    <div className="flex justify-between items-end mb-2">
                      <h4 className="text-emerald-400 font-bold tracking-wide">1. Vector Embeddings (FAISS HNSW)</h4>
                      <span className={`text-2xl font-mono ${getScoreColor(domainScore)}`}>{domainScore}</span>
                    </div>
                    <div className="bg-black/50 p-3 rounded font-mono text-xs text-gray-300 border border-gray-800 overflow-x-auto">
                      <span className="text-gray-500"># Cosine Similarity between 768-dimensional Vectors</span><br />
                      score = cosine_similarity([Resume_Vector], [JD_Vector]) * 100 = {domainScore}
                    </div>
                    <div className={`text-sm mt-2 font-medium ${getScoreColor(domainScore)}`}>
                      ↳ {domainScore >= 80 ? "Strong semantic alignment with the job description." : domainScore >= 50 ? "Moderate semantic match. Consider aligning your phrasing closer to the JD." : "Poor semantic match. The overall theme of your resume does not fit this role."}
                    </div>
                  </div>

                  {/* Experience */}
                  <div>
                    <div className="flex justify-between items-end mb-2">
                      <h4 className="text-purple-400 font-bold tracking-wide">2. Experience Penalty (Interval Merging)</h4>
                      <span className={`text-2xl font-mono ${getScoreColor(expFitScore)}`}>{expFitScore}</span>
                    </div>
                    <div className="bg-black/50 p-3 rounded font-mono text-xs text-gray-300 border border-gray-800 overflow-x-auto">
                      <span className="text-gray-500"># Calculated via Regex gap extraction</span><br />
                      gap = {jd_yoe} - {resume_yoe} = {Math.max(0, jd_yoe - resume_yoe).toFixed(1)}<br />
                      percent_miss = {Math.max(0, jd_yoe - resume_yoe).toFixed(1)} / {jd_yoe || 1}<br />
                      penalty = min(100.0, {yoe_penalty}) = {yoe_penalty}<br />
                      exp_fit_score = 100.0 - {yoe_penalty} = {expFitScore}
                    </div>
                    <div className={`text-sm mt-2 font-medium ${getScoreColor(expFitScore)}`}>
                      ↳ {yoe_penalty > 0 ? `Reduced by ${yoe_penalty} points: You are missing ${Math.max(0, jd_yoe - resume_yoe).toFixed(1)} years of the ${jd_yoe} required.` : "No penalty: You meet or exceed the required years of experience!"}
                    </div>
                  </div>

                  {/* Hard Keyword Score */}
                  <div>
                    <div className="flex justify-between items-end mb-2">
                      <h4 className="text-blue-400 font-bold tracking-wide">3. Hard Keyword Score (Categorical Weighting)</h4>
                      <span className={`text-2xl font-mono ${getScoreColor(skScore)}`}>{skScore}</span>
                    </div>
                    <div className="bg-black/50 p-3 rounded font-mono text-xs text-gray-300 border border-gray-800 overflow-x-auto">
                      <span className="text-gray-500"># Exact/Fuzzy match (Required=3x, Preferred=1x)</span><br />
                      score = {skScore} / 100
                    </div>
                    <div className={`text-sm mt-2 font-medium ${getScoreColor(skScore)}`}>
                      ↳ {skScore < 100 ? `Reduced by ${(100 - skScore).toFixed(1)} points due to missing high-value (IDF) skills.` : "Perfect score! You hit every high-value keyword."}
                    </div>
                  </div>

                  {/* Format */}
                  <div>
                    <div className="flex justify-between items-end mb-2">
                      <h4 className="text-orange-400 font-bold tracking-wide">4. Format / Section Completeness</h4>
                      <span className="text-2xl font-mono text-gray-400">{formatScore}</span>
                    </div>
                    <div className="bg-black/50 p-3 rounded font-mono text-xs text-gray-300 border border-gray-800 overflow-x-auto">
                      <span className="text-gray-500"># Display-only metric. Excluded from overall_score.</span><br />
                      score = ({found_sections} / 4) * 100 = {formatScore}
                    </div>
                  </div>
                </div>
              </div>

              {/* Panel 3: Skills Radar */}
              <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 shadow-2xl flex flex-col">
                <h3 className="text-xl font-bold mb-6 text-blue-400">LLM</h3>

                <div className="bg-black/50 p-3 rounded font-mono text-xs text-blue-300 border border-gray-800 mb-6">
                  # Scanned directly via Regex to Postgres!
                </div>

                <div className="flex-1 space-y-6">
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-3 uppercase tracking-wider">Matched JD Skills</h4>
                    <div className="space-y-2 max-h-40 overflow-y-auto pr-2">
                      {!selectedMatch || !selectedMatch.score_data?.match_details || selectedMatch.score_data.match_details.length === 0 ? (
                        <div className="text-sm text-gray-600 italic">No skills matched.</div>
                      ) : (
                        selectedMatch.score_data.match_details
                          .filter((m: any) => m.status === 'matched')
                          .map((match: any, i: number) => (
                            <div key={i} className="flex justify-between items-center border-b border-gray-800/50 pb-2">
                              <span className="font-mono text-sm text-emerald-400 flex items-center gap-2"><span className="w-1.5 h-1.5 bg-emerald-400 rounded-full"></span>{match.skill}</span>
                            </div>
                          ))
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-3 uppercase tracking-wider">Missing Required Skills (High IDF)</h4>
                    <div className="space-y-2 max-h-40 overflow-y-auto pr-2">
                      {missingRequiredSkills.length === 0 ? (
                        <div className="text-sm text-gray-600 italic">No required skills missing!</div>
                      ) : (
                        missingRequiredSkills.map((skill: string, i: number) => (
                          <div key={i} className="flex justify-between items-center border-b border-gray-800/50 pb-2">
                            <span className="font-mono text-sm text-red-500 flex items-center gap-2"><span className="w-1.5 h-1.5 bg-red-500 rounded-full"></span>{skill}</span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-3 uppercase tracking-wider">Missing Preferred Skills (Bonus)</h4>
                    <div className="space-y-2 max-h-40 overflow-y-auto pr-2">
                      {missingPreferredSkills.length === 0 ? (
                        <div className="text-sm text-gray-600 italic">No preferred skills missing!</div>
                      ) : (
                        missingPreferredSkills.map((skill: string, i: number) => (
                          <div key={i} className="flex justify-between items-center border-b border-gray-800/50 pb-2">
                            <span className="font-mono text-sm text-yellow-500 flex items-center gap-2"><span className="w-1.5 h-1.5 bg-yellow-500 rounded-full"></span>{skill}</span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Panel 4: Automated Insights (Zero-LLM) */}
              <div className="lg:col-span-3 bg-gray-900/50 border border-gray-800 rounded-xl p-6 shadow-2xl">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                  Automated Insights (Generated without AI)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-emerald-950/30 border border-emerald-900/50 p-5 rounded-lg">
                    <h4 className="text-emerald-400 font-bold mb-2 flex items-center gap-2">🟢 Why The Score Is High</h4>
                    <p className="text-sm text-gray-300 leading-relaxed">
                      {extractedSkills.length > 0
                        ? `You successfully matched ${extractedSkills.length} core technical requirements. The system detected strong alignment with high-value skills such as: ${extractedSkills.slice(0, 4).join(', ')}. Your resume's vector embeddings also indicate a ${domainScore >= 70 ? 'strong' : 'moderate'} overall semantic fit for this role.`
                        : "No major strengths detected. You did not match any of the required technical skills."}
                    </p>
                  </div>
                  <div className="bg-red-950/30 border border-red-900/50 p-5 rounded-lg">
                    <h4 className="text-red-400 font-bold mb-2 flex items-center gap-2">🔴 Where Points Were Lost</h4>
                    <p className="text-sm text-gray-300 leading-relaxed">
                      {missingSkills.length > 0
                        ? `Your hard-keyword score was penalized because the scanner could not find ${missingSkills.length} requested skills. The most critical missing requirements are: ${missingSkills.slice(0, 4).join(', ')}.`
                        : "Excellent! You did not lose any points for missing skills."}
                      {yoe_penalty > 0 && ` Additionally, you lost ${yoe_penalty} points because the JD requires ${jd_yoe} years of experience, but your timeline only demonstrates ${resume_yoe} years.`}
                    </p>
                  </div>
                </div>
              </div>

              {/* Panel 5: Raw Texts */}
              <div className="lg:col-span-3 bg-gray-900/50 border border-gray-800 rounded-xl p-6 shadow-2xl">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-xl font-bold mb-4">Raw Resume Extractor Output</h3>
                    <div className="bg-[#0d1117] p-4 rounded-lg font-mono text-xs text-gray-300 overflow-x-auto border border-gray-800 shadow-inner max-h-64 overflow-y-auto whitespace-pre-wrap">
                      {previewText}
                    </div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-4">Target Job Description</h3>
                    <div className="bg-[#0d1117] p-4 rounded-lg font-mono text-xs text-blue-300 overflow-x-auto border border-gray-800 shadow-inner max-h-64 overflow-y-auto whitespace-pre-wrap">
                      {selectedMatch?.full_jd_text || "JD text not available"}
                    </div>
                  </div>
                </div>
              </div>
            </div>



            {/* Panel 7: Gap Analysis (Module S4) */}
            {overallScore >= 75 && (
              <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 shadow-2xl mt-8">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-xl font-bold flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-orange-500 animate-pulse"></span>
                    Module S4: Gap Analysis
                  </h3>
                  {!gapAnalysisData && !gapAnalysisLoading && (
                    <button
                      onClick={handleGapAnalysis}
                      className="px-4 py-2 bg-orange-600 hover:bg-orange-700 font-bold rounded transition-colors text-sm"
                    >
                      Run Gap Analysis
                    </button>
                  )}
                </div>

                {gapAnalysisLoading && (
                  <div className="flex flex-col items-center justify-center py-8 space-y-4">
                    <div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-orange-400 font-mono text-sm animate-pulse">Analyzing bridgeable gaps...</p>
                  </div>
                )}

                {gapAnalysisData && (
                  <div className="space-y-6">
                    {/* Critical Gaps */}
                    <div>
                      <h4 className="text-red-400 font-bold mb-3 uppercase tracking-wider text-sm">Critical Gaps</h4>
                      {gapAnalysisData.gaps?.filter((g: any) => g.severity === 'CRITICAL').length === 0 ? (
                        <p className="text-gray-500 italic text-sm">No critical gaps found.</p>
                      ) : (
                        <ul className="list-disc pl-5 space-y-1 text-sm text-gray-300">
                          {gapAnalysisData.gaps.filter((g: any) => g.severity === 'CRITICAL').map((g: any, i: number) => (
                            <li key={i}><span className="font-bold text-gray-200">{g.skill}</span>: {g.context}</li>
                          ))}
                        </ul>
                      )}
                    </div>

                    {/* Bridgeable Gaps */}
                    <div>
                      <h4 className="text-emerald-400 font-bold mb-3 uppercase tracking-wider text-sm">Bridgeable Gaps</h4>
                      {gapAnalysisData.bridgeable_gaps?.length === 0 ? (
                        <p className="text-gray-500 italic text-sm">No bridgeable gaps identified.</p>
                      ) : (
                        <div className="space-y-3">
                          {gapAnalysisData.bridgeable_gaps.map((bg: any, i: number) => (
                            <div key={i} className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                              <div className="flex justify-between items-start mb-2">
                                <span className="font-bold text-emerald-300">{bg.missing_skill}</span>
                                <span className="text-xs bg-emerald-900/50 text-emerald-400 px-2 py-1 rounded">Adjacent: {bg.adjacent_skill}</span>
                              </div>
                              <p className="text-sm text-gray-300">{bg.rationale}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
}

export default App;
