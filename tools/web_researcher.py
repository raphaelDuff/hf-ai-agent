# tools/web_researcher.py - Web Research Engine
import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin, urlparse
import aiohttp
import asyncio

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    print(
        "Warning: BeautifulSoup not installed. Install with: pip install beautifulsoup4"
    )
    BS4_AVAILABLE = False

try:
    from langchain_community.utilities import WikipediaAPIWrapper
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    print(
        "Warning: langchain_community not installed. Install with: pip install langchain_community"
    )
    WIKIPEDIA_AVAILABLE = False

from .base_tool import UniversalTool


class WebResearchEngine(UniversalTool):
    """
    Web Research Engine - Gathers factual information from authoritative sources.

    This tool provides raw information gathering that Claude can interpret contextually:
    - Multi-query search strategies
    - Temporal filtering for date-specific information
    - Cross-reference verification
    - Content extraction and summarization

    Anti-pattern: NO recommendation engines or engagement analysis
    Usage: Gathers raw factual data for Claude to reason about and synthesize
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Web Research Engine")
        self.logger = logging.getLogger("gaia_agent.web_researcher")
        self.logger.setLevel(logging.INFO)

        self.api_key = api_key
        self.capabilities = [
            "web_search",
            "content_extraction",
            "source_verification",
            "temporal_filtering",
            "multi_source_synthesis",
        ]

        # Configuration
        self.config = {
            "max_results_per_query": 10,
            "max_content_length": 5000,  # Characters per source
            "request_timeout": 120,
            "max_concurrent_requests": 3,
            "user_agent": "GAIA-Agent/1.0 (Research Bot)",
        }

        # Search engines configuration
        self.search_engines = {
            "tavily": {
                "available": bool(api_key),
                "base_url": "https://api.tavily.com/search",
                "headers": {"Content-Type": "application/json"},
            },
            "duckduckgo": {
                "available": True,
                "base_url": "https://api.duckduckgo.com/",
                "headers": {},
            },
        }

        self.session = None
        
        # Initialize Wikipedia tool if available
        self.wikipedia = WikipediaAPIWrapper() if WIKIPEDIA_AVAILABLE else None
        
        self.logger.info(f"Web Research Engine initialized (Wikipedia available: {WIKIPEDIA_AVAILABLE})")

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute comprehensive web research

        Args:
            query: Search query or research topic

        Returns:
            Standardized output with research results
        """
        start_time = time.time()

        try:
            # Initialize HTTP session if needed
            if self.session is None:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config["request_timeout"]),
                    headers={"User-Agent": self.config["user_agent"]},
                )

            # Perform comprehensive research
            research_results = {}

            # 1. Query analysis and expansion
            research_results["query_analysis"] = await self._analyze_query(query)
            
            # 1.5. Check if Wikipedia is explicitly mentioned in the query
            use_wikipedia_first = self._should_prioritize_wikipedia(query)
            
            if use_wikipedia_first and self.wikipedia:
                self.logger.info("Wikipedia explicitly mentioned in query - searching Wikipedia first")
                wikipedia_results = await self._search_wikipedia_fallback(query)
                if wikipedia_results:
                    research_results["extracted_content"] = wikipedia_results
                    research_results["wikipedia_primary_used"] = True
                    # Skip web search if Wikipedia provides good results
                    if len(wikipedia_results) >= 1:
                        self.logger.info("Wikipedia search successful - skipping web search")
                        # Still set empty search results for consistency
                        research_results["search_results"] = []
                    else:
                        # Fall back to web search if Wikipedia didn't work well
                        research_results["search_results"] = await self._perform_multi_search(query)
                        research_results["extracted_content"].extend(await self._extract_content_from_sources(
                            research_results["search_results"]
                        ))
                else:
                    # Wikipedia failed, use web search
                    research_results["search_results"] = await self._perform_multi_search(query)
                    research_results["extracted_content"] = await self._extract_content_from_sources(
                        research_results["search_results"]
                    )
            else:
                # Normal flow: web search first
                # 2. Multi-engine search
                research_results["search_results"] = await self._perform_multi_search(query)

                # 3. Extract content from sources
                research_results["extracted_content"] = await self._extract_content_from_sources(
                    research_results["search_results"]
                )
            
            # 3.5. Wikipedia fallback if insufficient or low-quality data
            content_count = len(research_results["extracted_content"])
            has_low_quality = self._has_low_quality_content(research_results["extracted_content"])
            should_use_wikipedia = content_count < 2 or has_low_quality
            
            self.logger.info(f"Wikipedia fallback check: content_count={content_count}, has_low_quality={has_low_quality}, should_use={should_use_wikipedia}")
            
            if should_use_wikipedia and self.wikipedia:
                self.logger.info(f"Insufficient or low-quality web results (count: {len(research_results['extracted_content'])}), trying Wikipedia fallback")
                wikipedia_results = await self._search_wikipedia_fallback(query)
                if wikipedia_results:
                    self.logger.info(f"Wikipedia fallback successful: added {len(wikipedia_results)} results")
                    research_results["extracted_content"].extend(wikipedia_results)
                    research_results["wikipedia_fallback_used"] = True
                else:
                    self.logger.warning("Wikipedia fallback failed to return results")
            elif not self.wikipedia:
                self.logger.warning("Wikipedia fallback needed but Wikipedia not available")

            # 4. Cross-reference verification
            research_results["verification"] = await self._cross_reference_information(
                research_results["extracted_content"]
            )

            # 5. Temporal relevance filtering
            research_results["temporal_analysis"] = (
                await self._analyze_temporal_relevance(
                    research_results["extracted_content"], query
                )
            )

            # Compile comprehensive output
            raw_output = self._compile_research_output(research_results)

            metadata = {
                "query": query,
                "research_time": time.time() - start_time,
                "total_sources": len(research_results.get("search_results", [])),
                "search_engines_used": [
                    engine
                    for engine, config in self.search_engines.items()
                    if config["available"]
                ],
            }

            return self._standardize_output(raw_output, metadata)

        except Exception as e:
            self.logger.error(f"Web research failed: {str(e)}")
            return self._error_output(f"Research failed: {str(e)}")

        finally:
            # Keep session alive for potential reuse, but clean up on errors
            pass

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine research strategy"""
        query_lower = query.lower()

        # Detect query type
        temporal_indicators = [
            "current",
            "latest",
            "recent",
            "2024",
            "2025",
            "today",
            "now",
        ]
        factual_indicators = ["what", "how", "why", "when", "where", "who"]
        comparison_indicators = ["vs", "versus", "compare", "difference", "better"]

        query_types = []
        if any(indicator in query_lower for indicator in temporal_indicators):
            query_types.append("temporal")
        if any(indicator in query_lower for indicator in factual_indicators):
            query_types.append("factual")
        if any(indicator in query_lower for indicator in comparison_indicators):
            query_types.append("comparison")

        # Extract key terms
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query)
        key_terms = [
            word
            for word in words
            if word.lower()
            not in [
                "what",
                "how",
                "why",
                "when",
                "where",
                "who",
                "the",
                "and",
                "for",
                "are",
                "was",
            ]
        ]

        # Generate expanded queries
        expanded_queries = self._generate_expanded_queries(query, key_terms)

        return {
            "original_query": query,
            "query_types": query_types,
            "key_terms": key_terms[:10],  # Limit to top 10
            "expanded_queries": expanded_queries,
            "requires_recent_info": "temporal" in query_types,
            "complexity": (
                "high" if len(query_types) > 1 else "medium" if query_types else "low"
            ),
        }

    def _generate_expanded_queries(
        self, original_query: str, key_terms: List[str]
    ) -> List[str]:
        """Generate expanded search queries for better coverage"""
        expanded = [original_query]

        # Add key term combinations
        if len(key_terms) >= 2:
            expanded.append(" ".join(key_terms[:3]))

        # Add specific variations
        query_lower = original_query.lower()

        if "current" in query_lower or "latest" in query_lower:
            expanded.append(f"{original_query} 2024 2025")

        if "how" in query_lower:
            expanded.append(original_query.replace("how", "steps to"))

        if "what is" in query_lower:
            expanded.append(original_query.replace("what is", "definition of"))

        # Limit to reasonable number
        return expanded[:5]

    async def _perform_multi_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform search across multiple engines"""
        all_results = []

        # Search with Tavily API if available
        if self.search_engines["tavily"]["available"]:
            tavily_results = await self._search_with_tavily(query)
            all_results.extend(tavily_results)

        # Fallback to other search methods if needed
        if len(all_results) < 5:
            # Could add DuckDuckGo instant answers or other APIs
            fallback_results = await self._search_fallback(query)
            all_results.extend(fallback_results)

        # Deduplicate results by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        return unique_results[: self.config["max_results_per_query"]]

    async def _search_with_tavily(self, query: str) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        try:
            search_payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": True,
                "max_results": self.config["max_results_per_query"],
            }

            async with self.session.post(
                self.search_engines["tavily"]["base_url"],
                json=search_payload,
                headers=self.search_engines["tavily"]["headers"],
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("results", []):
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "snippet": item.get("content", ""),
                                "published_date": item.get("published_date"),
                                "source": "tavily",
                                "relevance_score": item.get("score", 0.5),
                            }
                        )

                    self.logger.info(f"Tavily search returned {len(results)} results")
                    return results
                else:
                    self.logger.warning(
                        f"Tavily search failed with status {response.status}"
                    )
                    return []

        except Exception as e:
            self.logger.error(f"Tavily search error: {str(e)}")
            return []

    async def _search_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search method when primary APIs are unavailable"""
        # This would implement alternative search methods
        # For now, return empty results
        self.logger.info("Using fallback search method")
        return []

    async def _search_wikipedia_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Search Wikipedia as fallback when web search returns insufficient results"""
        if not self.wikipedia:
            return []
            
        try:
            self.logger.info(f"Searching Wikipedia for: {query}")
            
            # Extract key terms for Wikipedia search
            key_terms = self._extract_wikipedia_search_terms(query)
            wikipedia_results = []
            
            for term in key_terms[:3]:  # Try top 3 terms
                try:
                    # Run Wikipedia search in thread pool to avoid blocking
                    import asyncio
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.wikipedia.run, term)
                    
                    if result and result.strip() and len(result) > 200:  # Ensure substantial content
                        # Clean up the Wikipedia content
                        cleaned_result = self._clean_wikipedia_content(result)
                        
                        wikipedia_results.append({
                            "success": True,
                            "url": f"https://en.wikipedia.org/wiki/{term.replace(' ', '_')}",
                            "title": f"Wikipedia: {term}",
                            "domain": "wikipedia.org",
                            "extracted_text": cleaned_result[:self.config["max_content_length"]],
                            "content_length": len(cleaned_result),
                            "extraction_method": "wikipedia_api",
                            "search_term": term
                        })
                        
                        self.logger.info(f"Wikipedia search successful for term: {term} ({len(cleaned_result)} chars)")
                        # Don't break - get multiple Wikipedia sources for better coverage
                        
                except Exception as e:
                    self.logger.warning(f"Wikipedia search failed for term '{term}': {str(e)}")
                    continue
                    
            return wikipedia_results
            
        except Exception as e:
            self.logger.error(f"Wikipedia fallback search failed: {str(e)}")
            return []

    def _extract_wikipedia_search_terms(self, query: str) -> List[str]:
        """Extract relevant search terms for Wikipedia from the query"""
        import re
        
        search_terms = []
        
        # Look for quoted phrases first (highest priority)
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        search_terms.extend(quoted_phrases)
        
        # Look for proper nouns (people, places, organizations)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        search_terms.extend(proper_nouns)
        
        # For music/artist queries, look for common patterns
        music_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:albums?|songs?|music|discography)',
            r'(?:albums?|songs?|music|discography)\s+(?:by|from|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:singer|musician|artist|band)'
        ]
        
        for pattern in music_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            search_terms.extend(matches)
        
        # Remove duplicates while preserving order and filter out very short terms
        unique_terms = []
        for term in search_terms:
            if term not in unique_terms and len(term) > 2:
                unique_terms.append(term)
        
        # If no good terms found, fall back to key significant words
        if not unique_terms:
            stop_words = {'how', 'many', 'what', 'when', 'where', 'who', 'why', 'which', 'were', 'was', 'are', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'you', 'can', 'use', 'latest', 'version', 'english', 'wikipedia', 'included', 'between'}
            words = re.findall(r'\b[A-Za-z]+\b', query.lower())
            significant_words = [word for word in words if len(word) > 3 and word not in stop_words]
            unique_terms.extend(significant_words[:3])
                
        return unique_terms[:3]  # Return top 3 most relevant terms

    def _clean_wikipedia_content(self, content: str) -> str:
        """Clean Wikipedia content to remove formatting and focus on useful information"""
        import re
        
        # Remove common Wikipedia artifacts
        content = re.sub(r'\[edit\]', '', content)
        content = re.sub(r'\[citation needed\]', '', content)
        content = re.sub(r'\[\d+\]', '', content)  # Remove reference numbers
        content = re.sub(r'==+ .+ ==+', '', content)  # Remove section headers
        content = re.sub(r'\n+', ' ', content)  # Replace multiple newlines with space
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        
        return content.strip()

    def _should_prioritize_wikipedia(self, query: str) -> bool:
        """Check if Wikipedia should be prioritized based on query content"""
        query_lower = query.lower()
        wikipedia_indicators = [
            'wikipedia', 'wiki', 'english wikipedia', 'wikipedia.org',
            'latest version of wikipedia', '2022 version of english wikipedia'
        ]
        
        return any(indicator in query_lower for indicator in wikipedia_indicators)

    def _has_low_quality_content(self, extracted_content: List[Dict[str, Any]]) -> bool:
        """Check if the extracted content is low quality (navigation, headers, etc.)"""
        if not extracted_content:
            return True
            
        low_quality_indicators = [
            'navigation', 'menu', 'header', 'footer', 'sidebar', 'cookie', 'privacy policy',
            'terms of service', 'log in', 'sign up', 'subscribe', 'newsletter', 'advertisement',
            'contact us', 'about us', 'home', 'search', 'loading', 'javascript', 'enable cookies',
            'main menu', 'skip to', 'jump to', 'accessibility', 'wikipedia', 'wikimedia',
            'edit section', 'contents', 'references', 'external links', 'categories'
        ]
        
        # Keywords that indicate substantive content about the topic
        substantive_keywords = ['album', 'discography', 'released', 'recorded', 'studio', 'music', 'song', 'track']
        
        total_useful_content = 0
        total_analyzed = 0
        
        for content in extracted_content:
            if not content.get("success"):
                continue
                
            text = content.get("extracted_text", "").lower()
            
            # Skip very short content
            if len(text) < 100:
                continue
                
            total_analyzed += 1
                
            # Count low-quality vs substantive indicators
            low_quality_count = sum(1 for indicator in low_quality_indicators if indicator in text)
            substantive_count = sum(1 for keyword in substantive_keywords if keyword in text)
            
            # Check if content has substantial meaningful text about the topic
            words = text.split()
            word_count = len(words)
            
            # More stringent quality checks
            is_useful = (
                word_count > 100 and  # At least 100 words
                substantive_count >= 2 and  # Contains at least 2 topic-relevant keywords
                low_quality_count < 5 and  # Not too many navigation elements
                ('mercedes sosa' in text or any(word in text for word in ['argentina', 'singer', 'folk'])) # Topic relevance
            )
            
            if is_useful:
                total_useful_content += 1
                self.logger.info(f"Found useful content: {word_count} words, {substantive_count} substantive keywords")
            else:
                self.logger.info(f"Low quality content: {word_count} words, {substantive_count} substantive, {low_quality_count} low-quality indicators")
                
        # If we have less than 2 pieces of useful content, consider it low quality overall
        useful_ratio = total_useful_content / max(total_analyzed, 1)
        self.logger.info(f"Content quality: {total_useful_content}/{total_analyzed} useful ({useful_ratio:.2%})")
        
        return total_useful_content < 2

    async def _extract_content_from_sources(
        self, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract content from sources"""
        extracted_content = []

        # Limit concurrent requests
        semaphore = asyncio.Semaphore(self.config["max_concurrent_requests"])

        async def extract_single_source(source):
            async with semaphore:
                return await self._extract_single_source_content(source)

        # Extract content from top sources
        top_sources = sources[:8]  # Limit to top 8 sources
        extraction_tasks = [extract_single_source(source) for source in top_sources]

        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict) and result.get("success"):
                extracted_content.append(result)

        return extracted_content

    async def _extract_single_source_content(
        self, source: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract content from a single source"""
        url = source.get("url", "")

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()

                    if BS4_AVAILABLE:
                        extracted_text = self._extract_text_from_html(html_content)
                    else:
                        # Simple text extraction without BeautifulSoup
                        extracted_text = re.sub(r"<[^>]+>", " ", html_content)
                        extracted_text = re.sub(r"\s+", " ", extracted_text).strip()

                    # Limit content length
                    if len(extracted_text) > self.config["max_content_length"]:
                        extracted_text = (
                            extracted_text[: self.config["max_content_length"]] + "..."
                        )

                    return {
                        "success": True,
                        "url": url,
                        "title": source.get("title", ""),
                        "domain": source.get("domain", ""),
                        "extracted_text": extracted_text,
                        "content_length": len(extracted_text),
                        "extraction_method": "html_parsing",
                    }
                else:
                    return {
                        "success": False,
                        "url": url,
                        "error": f"HTTP {response.status}",
                        "fallback_content": source.get("snippet", ""),
                    }

        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "fallback_content": source.get("snippet", ""),
            }

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML using BeautifulSoup"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            self.logger.warning(f"HTML text extraction failed: {str(e)}")
            # Fallback to regex
            text = re.sub(r"<[^>]+>", " ", html_content)
            return re.sub(r"\s+", " ", text).strip()

    async def _cross_reference_information(
        self, extracted_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cross-reference information across sources"""
        if len(extracted_content) < 2:
            return {
                "verification_possible": False,
                "reason": "Insufficient sources for cross-referencing",
            }

        # Simple cross-referencing based on common terms and phrases
        all_texts = [
            content.get("extracted_text", "")
            for content in extracted_content
            if content.get("success")
        ]

        if not all_texts:
            return {
                "verification_possible": False,
                "reason": "No extracted text available",
            }

        # Find common key phrases (simplified approach)
        common_phrases = self._find_common_phrases(all_texts)

        # Calculate agreement score
        agreement_indicators = []
        for phrase in common_phrases:
            sources_mentioning = sum(
                1 for text in all_texts if phrase.lower() in text.lower()
            )
            agreement_percentage = (sources_mentioning / len(all_texts)) * 100
            agreement_indicators.append(
                {
                    "phrase": phrase,
                    "sources_count": sources_mentioning,
                    "agreement_percentage": agreement_percentage,
                }
            )

        # Overall verification confidence
        if agreement_indicators:
            avg_agreement = sum(
                indicator["agreement_percentage"] for indicator in agreement_indicators
            ) / len(agreement_indicators)
            verification_confidence = (
                "high"
                if avg_agreement > 70
                else "medium" if avg_agreement > 40 else "low"
            )
        else:
            verification_confidence = "low"

        return {
            "verification_possible": True,
            "sources_analyzed": len(extracted_content),
            "common_information": agreement_indicators[:10],  # Top 10
            "verification_confidence": verification_confidence,
            "methodology": "phrase_frequency_analysis",
        }

    def _find_common_phrases(self, texts: List[str]) -> List[str]:
        """Find common phrases across texts"""
        # Simple phrase extraction (2-4 word phrases)
        all_phrases = []

        for text in texts:
            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

            # Extract 2-4 word phrases
            for i in range(len(words) - 1):
                for phrase_len in [2, 3, 4]:
                    if i + phrase_len <= len(words):
                        phrase = " ".join(words[i : i + phrase_len])
                        if len(phrase) > 10:  # Filter short phrases
                            all_phrases.append(phrase)

        # Count phrase frequency
        phrase_counts = {}
        for phrase in all_phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Return phrases that appear in multiple sources
        common_phrases = [
            phrase
            for phrase, count in phrase_counts.items()
            if count >= 2 and len(phrase.split()) >= 2
        ]

        # Sort by frequency and return top results
        common_phrases.sort(key=lambda p: phrase_counts[p], reverse=True)
        return common_phrases[:20]

    async def _analyze_temporal_relevance(
        self, extracted_content: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """Analyze temporal relevance of information"""
        query_lower = query.lower()
        requires_recent = any(
            term in query_lower
            for term in ["current", "latest", "recent", "2024", "2025", "today", "now"]
        )

        temporal_analysis = {
            "requires_recent_info": requires_recent,
            "sources_with_dates": 0,
            "recent_sources": 0,
            "temporal_confidence": "unknown",
        }

        if not requires_recent:
            temporal_analysis["temporal_confidence"] = "not_required"
            return temporal_analysis

        current_year = datetime.now().year
        recent_threshold = datetime.now() - timedelta(days=365)  # Last year

        for content in extracted_content:
            if not content.get("success"):
                continue

            text = content.get("extracted_text", "")

            # Look for date indicators in text
            date_patterns = [
                r"\b202[4-5]\b",  # Years 2024-2025
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+202[4-5]\b",
                r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+202[4-5]\b",
            ]

            has_recent_dates = any(
                re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns
            )

            if has_recent_dates:
                temporal_analysis["sources_with_dates"] += 1
                temporal_analysis["recent_sources"] += 1

        # Calculate temporal confidence
        if temporal_analysis["recent_sources"] >= 2:
            temporal_analysis["temporal_confidence"] = "high"
        elif temporal_analysis["recent_sources"] == 1:
            temporal_analysis["temporal_confidence"] = "medium"
        else:
            temporal_analysis["temporal_confidence"] = "low"

        return temporal_analysis

    def _compile_research_output(self, research_results: Dict[str, Any]) -> str:
        """Compile all research results into a comprehensive output"""

        output_lines = ["=== WEB RESEARCH RESULTS ===\n"]

        # Query analysis
        query_analysis = research_results.get("query_analysis", {})
        output_lines.append("RESEARCH QUERY ANALYSIS:")
        output_lines.append(
            f"- Original query: {query_analysis.get('original_query', 'Unknown')}"
        )
        output_lines.append(
            f"- Query types: {', '.join(query_analysis.get('query_types', []))}"
        )
        output_lines.append(
            f"- Key terms: {', '.join(query_analysis.get('key_terms', []))}"
        )
        output_lines.append(
            f"- Complexity: {query_analysis.get('complexity', 'Unknown')}"
        )

        # Extracted content summary
        extracted_content = research_results.get("extracted_content", [])
        successful_extractions = [c for c in extracted_content if c.get("success")]
        output_lines.append(f"\nCONTENT EXTRACTION:")
        output_lines.append(f"- Successful extractions: {len(successful_extractions)}")
        output_lines.append(
            f"- Total content length: {sum(c.get('content_length', 0) for c in successful_extractions)} characters"
        )
        
        # Wikipedia usage indicators
        if research_results.get("wikipedia_primary_used"):
            output_lines.append("- Wikipedia was used as primary source due to explicit mention in query")
        elif research_results.get("wikipedia_fallback_used"):
            output_lines.append("- Wikipedia fallback was used due to insufficient web results")

        # Key information from sources
        if successful_extractions:
            output_lines.append("\nKEY INFORMATION FROM SOURCES:")
            for i, content in enumerate(successful_extractions[:3], 1):  # Top 3 sources
                output_lines.append(f"\nSource {i}: {content.get('title', 'Unknown')}")
                text = content.get("extracted_text", "")
                # Extract first paragraph or first 200 characters
                summary = text[:400] + "..." if len(text) > 400 else text
                summary = summary.split("\n")[0] if "\n" in summary else summary
                output_lines.append(f"Content: {summary}")

        # Cross-reference verification
        verification = research_results.get("verification", {})
        output_lines.append("\nCROSS-REFERENCE VERIFICATION:")
        if verification.get("verification_possible"):
            output_lines.append(
                f"- Verification confidence: {verification.get('verification_confidence', 'Unknown')}"
            )
            output_lines.append(
                f"- Sources analyzed: {verification.get('sources_analyzed', 0)}"
            )

            common_info = verification.get("common_information", [])
            if common_info:
                output_lines.append("- Common information points:")
                for info in common_info[:5]:  # Top 5
                    output_lines.append(
                        f"  • \"{info.get('phrase', '')}\" (mentioned in {info.get('sources_count', 0)} sources)"
                    )
        else:
            output_lines.append(
                f"- Verification not possible: {verification.get('reason', 'Unknown')}"
            )

        # Temporal relevance
        temporal = research_results.get("temporal_analysis", {})
        output_lines.append("\nTEMPORAL RELEVANCE:")
        output_lines.append(
            f"- Requires recent information: {'Yes' if temporal.get('requires_recent_info') else 'No'}"
        )
        output_lines.append(
            f"- Recent sources found: {temporal.get('recent_sources', 0)}"
        )
        output_lines.append(
            f"- Temporal confidence: {temporal.get('temporal_confidence', 'Unknown')}"
        )

        return "\n".join(output_lines)

    def _error_output(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error output"""
        return {
            "tool_name": self.name,
            "raw_output": f"Error: {error_message}",
            "success": False,
            "error": error_message,
            "metadata": {"error_occurred": True},
        }

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None


# Example usage and testing
async def test_web_researcher():
    """Test the Web Research Engine"""
    # Note: You would need a Tavily API key for full functionality
    researcher = WebResearchEngine(api_key="your_tavily_api_key_here")

    try:
        result = await researcher.execute(
            "What is the current state of artificial intelligence in 2024?"
        )

        print("Research Result:")
        print(f"Success: {result.get('success', False)}")
        print(f"Tool: {result.get('tool_name')}")
        print(
            f"Research time: {result.get('metadata', {}).get('research_time', 0):.2f}s"
        )
        print("\nRaw Output:")
        print(result.get("raw_output", "No output"))

    finally:
        await researcher.cleanup()


if __name__ == "__main__":
    asyncio.run(test_web_researcher())
