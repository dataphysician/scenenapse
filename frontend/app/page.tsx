"use client"

import React, { useState, useRef, useEffect } from 'react';
import {
  Layers, Box, Camera, Activity, Paperclip, Wand2, X, Plus,
  Save, ArrowLeft, Info, Edit2, Trash2, FileJson, Film,
  Aperture, User, FileText, Image as ImageIcon, Video, Mic,
  Zap, Layout, Palette, Target, Search, Terminal, Menu, Sparkles, HelpCircle,
  ChevronLeft, ChevronRight
} from 'lucide-react';
import {
  Scene, SceneResponse, RefineResponse, Element, Elements, SceneObject, Action, Cinematography,
  SceneState, CategoryType, ChatMessage, Attachment,
  DEFAULT_CINEMATOGRAPHY, SHOT_SIZES, CAMERA_ANGLES, LIGHTING_CONDITIONS, ARTISTIC_STYLES
} from '@/lib/types';

// ============================================================
// Initial State
// ============================================================

const INITIAL_SCENE: SceneState = {
  elements: { elements: [] },
  objects: { objects: [] },
  actions: { actions: [] },
  cinematography: null
};

// ============================================================
// Backend Sync API Functions
// ============================================================

async function fetchSceneState(): Promise<SceneState | null> {
  try {
    const response = await fetch('/api/scene');
    if (!response.ok) return null;
    const data = await response.json();
    return {
      elements: data.elements,
      objects: { objects: data.objects },
      actions: { actions: data.actions },
      cinematography: data.cinematography,
    };
  } catch (error) {
    console.error('Failed to fetch scene state:', error);
    return null;
  }
}

async function syncElements(elements: Element[]): Promise<boolean> {
  try {
    const response = await fetch('/api/scene/elements', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ elements }),
    });
    return response.ok;
  } catch (error) {
    console.error('Failed to sync elements:', error);
    return false;
  }
}

async function syncObjects(objects: SceneObject[]): Promise<boolean> {
  try {
    const response = await fetch('/api/scene/objects', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ objects }),
    });
    return response.ok;
  } catch (error) {
    console.error('Failed to sync objects:', error);
    return false;
  }
}

async function syncActions(actions: Action[]): Promise<boolean> {
  try {
    const response = await fetch('/api/scene/actions', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actions }),
    });
    return response.ok;
  } catch (error) {
    console.error('Failed to sync actions:', error);
    return false;
  }
}

async function syncCinematography(cinematography: Cinematography | null): Promise<boolean> {
  try {
    const response = await fetch('/api/scene/cinematography', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cinematography }),
    });
    return response.ok;
  } catch (error) {
    console.error('Failed to sync cinematography:', error);
    return false;
  }
}

async function clearBackendScene(): Promise<boolean> {
  try {
    const response = await fetch('/api/scene', { method: 'DELETE' });
    return response.ok;
  } catch (error) {
    console.error('Failed to clear scene:', error);
    return false;
  }
}

// ============================================================
// Main Page Component
// ============================================================

export default function Home() {
  // State
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sceneData, setSceneData] = useState<SceneState>(INITIAL_SCENE);
  const [shortDescription, setShortDescription] = useState('');
  const [criticScore, setCriticScore] = useState(0);
  const [criticIssues, setCriticIssues] = useState<string[]>([]);
  const [syncStatus, setSyncStatus] = useState<'synced' | 'syncing' | 'error'>('synced');

  // UI State
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeCategory, setActiveCategory] = useState<CategoryType>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Refinement mode: After initial assembly, scene is configured and future inputs refine it
  const [isSceneConfigured, setIsSceneConfigured] = useState(false);

  // Pipeline logs for verbose sidebar display
  const [pipelineLogs, setPipelineLogs] = useState<{
    id: string;
    timestamp: number;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
    issues?: string[];
  }[]>([]);
  const [isPipelineActive, setIsPipelineActive] = useState(false);

  // Cancellation support for pipeline (Escape key)
  const pipelineCancelledRef = useRef(false);
  const [isImageGenerating, setIsImageGenerating] = useState(false);

  // Generated images state (shared with sidebar)
  const [generatedImages, setGeneratedImages] = useState<Array<{url: string; width: number; height: number; quality_score?: number | null}>>([]);
  const [selectedGeneratedId, setSelectedGeneratedId] = useState<number | null>(null);
  const [imageModalOpen, setImageModalOpen] = useState(false);
  const [imageModalIndex, setImageModalIndex] = useState(0);

  // Reference images state (Freepik search)
  const [referenceImages, setReferenceImages] = useState<Array<{id: number; url: string; thumbnail: string | null; title: string | null}>>([]);
  const [isSearchingImages, setIsSearchingImages] = useState(false);

  // About modal state
  const [showAboutModal, setShowAboutModal] = useState(false);
  const [aboutCarouselIndex, setAboutCarouselIndex] = useState(0);
  const aboutImages = ['/workflow-infographic.jpeg', '/workflow-infographic2.jpeg'];

  // References sidebar state (separate from main sidebar)
  const [showReferencesSidebar, setShowReferencesSidebar] = useState(false);
  const [selectedRefId, setSelectedRefId] = useState<number | null>(null);

  // Sync reference and generated image state from backend (for LLM-driven updates)
  const syncReferenceState = async () => {
    try {
      const response = await fetch('/api/scene');
      if (response.ok) {
        const data = await response.json();
        // Update reference images if backend has them
        if (data.reference_images && Array.isArray(data.reference_images)) {
          setReferenceImages(data.reference_images);
          // Also open the sidebar if we have images
          if (data.reference_images.length > 0) {
            setShowReferencesSidebar(true);
          }
        }
        // Update selected reference
        if (data.selected_reference) {
          const selectedImg = data.selected_reference;
          setSelectedRefId(selectedImg.id);
          // Update attachment
          const newAttachment: Attachment = {
            id: `ref-${selectedImg.id}`,
            type: 'image',
            name: selectedImg.title || `Reference ${selectedImg.id}`,
            data: selectedImg.thumbnail || selectedImg.url,
            mimeType: 'image/jpeg',
          };
          setAttachments(prev => [...prev.filter(a => !a.id.startsWith('ref-')), newAttachment]);
        } else {
          setSelectedRefId(null);
          setAttachments(prev => prev.filter(a => !a.id.startsWith('ref-')));
        }

        // Update generated images if backend has them
        if (data.generated_images && Array.isArray(data.generated_images)) {
          const imgs = data.generated_images.map((img: { url: string; width: number; height: number }) => ({
            url: img.url,
            width: img.width,
            height: img.height,
          }));
          if (imgs.length > 0 && generatedImages.length === 0) {
            setGeneratedImages(imgs);
          }
        }
        // Update selected generated
        if (data.selected_generated) {
          setSelectedGeneratedId(data.selected_generated.index);
        }
      }
    } catch (error) {
      console.error('Failed to sync state:', error);
    }
  };

  // Toggle reference image selection (single selection only) - calls backend API
  const handleToggleRefSelect = async (img: {id: number; url: string; thumbnail: string | null; title: string | null}) => {
    // Find the 1-based index of this image
    const index = referenceImages.findIndex(r => r.id === img.id) + 1;
    if (index < 1) return;

    try {
      const response = await fetch('/api/select-reference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.selected) {
          // Selected
          setSelectedRefId(data.selected.id);
          const newAttachment: Attachment = {
            id: `ref-${data.selected.id}`,
            type: 'image',
            name: data.selected.title || `Reference ${data.selected.id}`,
            data: data.selected.thumbnail || data.selected.url,
            mimeType: 'image/jpeg',
          };
          setAttachments(prev => [...prev.filter(a => !a.id.startsWith('ref-')), newAttachment]);
        } else {
          // Deselected
          setSelectedRefId(null);
          setAttachments(prev => prev.filter(a => !a.id.startsWith('ref-')));
        }
      }
    } catch (error) {
      console.error('Failed to select reference:', error);
    }
  };

  // Deselect reference image - calls backend API
  const handleDeselectReference = async () => {
    try {
      const response = await fetch('/api/deselect-reference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        setSelectedRefId(null);
        // Attachment removal is handled by the caller
      }
    } catch (error) {
      console.error('Failed to deselect reference:', error);
    }
  };

  // Toggle generated image selection - calls backend API
  const handleSelectGenerated = async (index: number) => {
    try {
      const response = await fetch('/api/select-generated', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.selected) {
          setSelectedGeneratedId(data.selected.index);
        } else {
          setSelectedGeneratedId(null);
        }
      }
    } catch (error) {
      console.error('Failed to select generated image:', error);
    }
  };

  // Open image modal at specific index
  const handleOpenImageModal = (index: number) => {
    setImageModalIndex(index);
    setImageModalOpen(true);
  };

  // Close modal and select the currently viewed image
  const handleCloseImageModal = () => {
    const imageIndex = imageModalIndex + 1; // Convert to 1-based
    handleSelectGenerated(imageIndex);
    setImageModalOpen(false);
  };

  // Poll for reference state changes when sidebar is open (for LLM-driven updates)
  useEffect(() => {
    if (!showReferencesSidebar) return;

    // Poll every 2 seconds when sidebar is open
    const interval = setInterval(syncReferenceState, 2000);
    return () => clearInterval(interval);
  }, [showReferencesSidebar]);

  // Helper to add a log entry and open sidebar
  const addPipelineLog = (message: string, type: 'info' | 'success' | 'warning' | 'error', issues?: string[]) => {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      message,
      type,
      issues,
    };
    setPipelineLogs(prev => [...prev, entry]);
  };

  // Start pipeline logging (clears previous logs and opens sidebar)
  const startPipelineLogging = () => {
    setPipelineLogs([]);
    setIsPipelineActive(true);
    setActiveCategory('assembled');
    setSidebarOpen(true);
  };

  // End pipeline logging (keeps logs visible until next pipeline or manual close)
  const endPipelineLogging = (success: boolean) => {
    setIsPipelineActive(false);
    // Logs remain visible in sidebar showing the transformation history
  };

  // Listen for Escape key to cancel pipeline
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && (isPipelineActive || isImageGenerating)) {
        pipelineCancelledRef.current = true;
        addPipelineLog('Pipeline cancelled by user (Escape pressed)', 'warning');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPipelineActive, isImageGenerating]);

  // Fetch initial scene state from backend on mount
  useEffect(() => {
    const loadInitialState = async () => {
      const state = await fetchSceneState();
      if (state && state.elements.elements.length > 0) {
        setSceneData(state);
      }
    };
    loadInitialState();
  }, []);

  const handleOpenSidebar = (category: CategoryType) => {
    setActiveCategory(category);
    setSidebarOpen(true);
  };

  const handleCloseSidebar = () => {
    setSidebarOpen(false);
    setTimeout(() => setActiveCategory(null), 300);
  };

  const handleClearCategory = async (category: CategoryType) => {
    if (category === 'cinematography') {
      setSceneData(prev => ({ ...prev, cinematography: null }));
      await syncCinematography(null);
    } else if (category === 'objects') {
      setSceneData(prev => ({ ...prev, objects: { objects: [] } }));
      await syncObjects([]);
    } else if (category === 'actions') {
      setSceneData(prev => ({ ...prev, actions: { actions: [] } }));
      await syncActions([]);
    } else if (category === 'elements') {
      // Clearing elements resets the entire scene and exits refinement mode
      setSceneData(prev => ({
        ...prev,
        elements: { elements: [] },
        objects: { objects: [] },
        actions: { actions: [] },
        cinematography: null
      }));
      setIsSceneConfigured(false);  // Exit refinement mode
      setShortDescription('');
      setCriticScore(0);
      setCriticIssues([]);
      await clearBackendScene();
    }
  };

  const handleUpdateSceneData = async (category: CategoryType, data: any) => {
    const stateKey = category === 'elements' ? 'elements' : category;
    setSceneData(prev => ({
      ...prev,
      [stateKey as string]: data
    }));

    // Sync with backend
    setSyncStatus('syncing');
    let success = false;

    if (category === 'elements') {
      success = await syncElements(data.elements);
    } else if (category === 'objects') {
      success = await syncObjects(data.objects);
    } else if (category === 'actions') {
      success = await syncActions(data.actions);
    } else if (category === 'cinematography') {
      success = await syncCinematography(data);
    }

    setSyncStatus(success ? 'synced' : 'error');
  };

  // Refinement pipeline (existing scene + instruction → classifier → targeted updates)
  // Used when there's both manual UI input AND semantic input (text/files)
  const runRefinePipeline = async (instruction: string, currentAttachments: Attachment[]) => {
    const CRITIC_THRESHOLD = 0.85;

    // Start logging to sidebar
    startPipelineLogging();
    addPipelineLog('Starting refinement pipeline...', 'info');

    // Stage 1: Fetch current scene and classify instruction
    addPipelineLog('Classifying refinement instruction...', 'info');

    try {
      // Fetch assembled scene from backend
      const sceneResponse = await fetch('/api/scene/assembled', {
        cache: 'no-store',
      });

      if (!sceneResponse.ok) {
        throw new Error('Failed to fetch assembled scene');
      }

      const scene = await sceneResponse.json();
      addPipelineLog('Fetched current scene from backend', 'info');

      // Build instruction - image context will be added by backend if image provided
      let fullInstruction = instruction || 'Refine the scene based on the attached reference image';
      if (currentAttachments.length > 0) {
        const attachmentTypes = currentAttachments.map(a => a.type).join(', ');
        addPipelineLog(`Processing attachments: ${attachmentTypes}`, 'info');
      }

      // Stage 2: Show refinement in progress with progress timers
      addPipelineLog('Analyzing affected components...', 'info');
      addPipelineLog('Running Best-of-N refinement candidates in parallel...', 'info');

      const progressTimers: NodeJS.Timeout[] = [];

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Parsing refinement instruction...', 'info');
      }, 2000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Updating affected scene heads...', 'info');
      }, 5000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Running critic validation on refinement...', 'info');
      }, 10000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Evaluating refinement quality...', 'info');
      }, 15000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Selecting best refinement candidate...', 'info');
      }, 20000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Regenerating scene summary...', 'info');
      }, 25000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Still refining... Complex changes may take longer.', 'info');
      }, 35000));

      // Build FormData for refinement (supports multimodal uploads)
      const formData = new FormData();
      formData.append('scene_json', JSON.stringify(scene));
      formData.append('instruction', fullInstruction);
      formData.append('regenerate_summary', 'true');

      // Add multimodal attachments if present
      for (const attachment of currentAttachments) {
        if (attachment.file && (attachment.type === 'image' || attachment.type === 'audio')) {
          formData.append(attachment.type, attachment.file);
          addPipelineLog(`Uploading reference ${attachment.type}: ${attachment.name}`, 'info');
        }
      }

      // Call refinement endpoint with FormData
      const response = await fetch('/api/refine', {
        method: 'POST',
        body: formData,
        cache: 'no-store',
      });

      // Clear all progress timers
      progressTimers.forEach(timer => clearTimeout(timer));

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        const errorMessage = errorBody?.detail
          || (typeof errorBody === 'string' ? errorBody : null)
          || `API error: ${response.status}`;
        throw new Error(errorMessage);
      }

      const data: RefineResponse = await response.json();

      // Stage 3: Show which heads were affected
      const affectedHeadsDisplay = data.affected_heads.length > 0
        ? data.affected_heads.map(h => h.toLowerCase()).join(', ')
        : 'all components';

      addPipelineLog(`Affected heads: ${affectedHeadsDisplay}`, 'info');

      // Stage 4: Show retry info if retries occurred
      const criticPassed = data.critic_score >= CRITIC_THRESHOLD;

      if (data.retry_count > 0) {
        addPipelineLog(`Best-of-N selection complete. Critic score below threshold.`, 'warning');
        addPipelineLog(`Retried refinement ${data.retry_count} time${data.retry_count > 1 ? 's' : ''} due to low confidence`, 'warning', data.critic_issues);
      } else if (!criticPassed && data.critic_issues && data.critic_issues.length > 0) {
        addPipelineLog('Alignment issues detected after Best-of-N selection', 'warning', data.critic_issues);
      } else {
        addPipelineLog('Best-of-N selection complete. Critic validation passed.', 'success');
      }

      // Stage 5: Summary generation
      addPipelineLog('Generating Best-of-N summary...', 'info');

      // Update state with refined scene
      const newSceneState: SceneState = {
        elements: data.scene.elements,
        objects: { objects: data.scene.objects },
        actions: { actions: data.scene.actions },
        cinematography: data.scene.cinematography,
      };

      setSceneData(newSceneState);
      if (data.short_description) {
        setShortDescription(data.short_description);
      }
      setCriticScore(data.critic_score);
      setCriticIssues(data.critic_issues || []);

      // Stage 6: Complete
      addPipelineLog(
        criticPassed
          ? `Refinement complete: ${affectedHeadsDisplay} (accuracy: ${(data.critic_score * 100).toFixed(0)}%)`
          : `Refinement complete with issues: ${affectedHeadsDisplay} (accuracy: ${(data.critic_score * 100).toFixed(0)}%)`,
        criticPassed ? 'success' : 'warning'
      );

      endPipelineLogging(criticPassed);

      return { data, newSceneState };

    } catch (error) {
      console.error('Refine error:', error);
      addPipelineLog(`Error: ${error instanceof Error ? error.message : 'Refinement failed'}`, 'error');
      endPipelineLogging(false);
      throw error;
    }
  };

  // Full generation pipeline (prompt → elements → all heads → critic → summary)
  // Works for both semantic input (text/files) and manual mode (stringified Scene)
  const runGeneratePipeline = async (prompt: string, currentAttachments: Attachment[]) => {
    const CRITIC_THRESHOLD = 0.85;

    // Start logging to sidebar
    startPipelineLogging();
    addPipelineLog('Starting generation pipeline...', 'info');

    // Stage 1: Analyzing prompt
    addPipelineLog('Analyzing prompt...', 'info');

    try {
      addPipelineLog('Extracting elements from prompt...', 'info');

      // Build form data
      const formData = new FormData();
      formData.append('prompt', prompt);

      for (const attachment of currentAttachments) {
        if (attachment.file) {
          formData.append(attachment.type, attachment.file);
          addPipelineLog(`Processing ${attachment.type} attachment: ${attachment.name}`, 'info');
        }
      }

      // Stage 2: Show progress updates via logging
      addPipelineLog('Running Best-of-N generation candidates in parallel...', 'info');

      const progressTimers: NodeJS.Timeout[] = [];

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Stage 1: Extracting scene elements...', 'info');
      }, 2000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Stage 2: Generating objects, actions, cinematography...', 'info');
      }, 5000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Running critic validation on candidates...', 'info');
      }, 10000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Evaluating candidate quality scores...', 'info');
      }, 15000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Selecting best candidate from pool...', 'info');
      }, 20000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Stage 3: Generating scene summary...', 'info');
      }, 25000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Still processing... Large scenes may take longer.', 'info');
      }, 35000));

      progressTimers.push(setTimeout(() => {
        addPipelineLog('Almost there... Finalizing scene structure.', 'info');
      }, 50000));

      const response = await fetch('/api/generate', {
        method: 'POST',
        body: formData,
        cache: 'no-store',
      });

      // Clear all progress timers
      progressTimers.forEach(timer => clearTimeout(timer));

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        const errorMessage = errorBody?.detail
          || (typeof errorBody === 'string' ? errorBody : null)
          || `API error: ${response.status}`;
        throw new Error(errorMessage);
      }

      const data: SceneResponse = await response.json();
      const criticPassed = data.critic_score >= CRITIC_THRESHOLD;

      // Stage 4: Show retry info if retries occurred
      if (data.retry_count > 0) {
        addPipelineLog('Best-of-N selection complete. Critic score below threshold.', 'warning');
        addPipelineLog(`Retried generation ${data.retry_count} time${data.retry_count > 1 ? 's' : ''} due to low confidence`, 'warning', data.critic_issues);
      } else if (!criticPassed && data.critic_issues && data.critic_issues.length > 0) {
        addPipelineLog('Alignment issues detected after Best-of-N selection', 'warning', data.critic_issues);
      } else {
        addPipelineLog('Best-of-N selection complete. Critic validation passed.', 'success');
      }

      // Stage 5: Summary
      addPipelineLog('Generating Best-of-N summary...', 'info');

      // Update state with results
      const newSceneState: SceneState = {
        elements: data.scene.elements,
        objects: { objects: data.scene.objects },
        actions: { actions: data.scene.actions },
        cinematography: data.scene.cinematography,
      };

      setSceneData(newSceneState);
      setShortDescription(data.short_description);
      setCriticScore(data.critic_score);
      setCriticIssues(data.critic_issues || []);

      // Final success message
      addPipelineLog(
        criticPassed
          ? `Generation complete (accuracy: ${(data.critic_score * 100).toFixed(0)}%)`
          : `Generation complete with issues (accuracy: ${(data.critic_score * 100).toFixed(0)}%)`,
        criticPassed ? 'success' : 'warning'
      );

      endPipelineLogging(criticPassed);

      return { data, newSceneState };

    } catch (error) {
      console.error('Generate error:', error);
      addPipelineLog(`Error: ${error instanceof Error ? error.message : 'Generation failed'}`, 'error');
      endPipelineLogging(false);
      throw error;
    }
  };

  // Generate images using Fal AI with pipeline logging - streams images as they complete via SSE
  const runImageGeneration = async (): Promise<Array<{url: string; width: number; height: number; quality_score?: number | null}>> => {
    // Check for cancellation before starting
    if (pipelineCancelledRef.current) {
      addPipelineLog('Image generation skipped (cancelled)', 'warning');
      return [];
    }

    // Hide references sidebar when generating
    setShowReferencesSidebar(false);

    setIsImageGenerating(true);
    setGeneratedImages([]); // Clear previous images
    setSelectedGeneratedId(null); // Clear selection
    addPipelineLog('Starting image generation with Fal AI...', 'info');
    addPipelineLog('Model: nano-banana-pro | Images: 4 | Resolution: 1K (streaming)', 'info');
    if (shortDescription) {
      addPipelineLog(`Prompt: "${shortDescription.substring(0, 100)}${shortDescription.length > 100 ? '...' : ''}"`, 'info');
    }

    return new Promise((resolve, reject) => {
      const images: Array<{url: string; width: number; height: number; quality_score?: number | null}> = [];
      const prompt = shortDescription || '';
      const url = `/api/generate-images-stream${prompt ? `?prompt=${encodeURIComponent(prompt)}` : ''}`;

      const eventSource = new EventSource(url);

      eventSource.onmessage = (event) => {
        // Check for cancellation
        if (pipelineCancelledRef.current) {
          eventSource.close();
          setIsImageGenerating(false);
          addPipelineLog('Image generation cancelled', 'warning');
          resolve(images);
          return;
        }

        const data = JSON.parse(event.data);

        if (data.error) {
          eventSource.close();
          setIsImageGenerating(false);
          addPipelineLog(`Error: ${data.error}`, 'error');
          reject(new Error(data.error));
          return;
        }

        if (data.done) {
          eventSource.close();
          setIsImageGenerating(false);
          addPipelineLog(`Generated ${images.length}/4 images`, 'success');
          resolve(images);
          return;
        }

        // New image received - stream it to UI
        const img = { url: data.url, width: data.width, height: data.height, quality_score: data.quality_score };
        images.push(img);
        setGeneratedImages(prev => [...prev, img]);
        const scoreText = data.quality_score != null ? ` (quality: ${(data.quality_score * 100).toFixed(0)}%)` : '';
        addPipelineLog(`Image ${images.length}/4 received${scoreText}`, 'success');
      };

      eventSource.onerror = () => {
        eventSource.close();
        setIsImageGenerating(false);
        if (images.length > 0) {
          addPipelineLog(`Completed with ${images.length} images`, 'warning');
          resolve(images);
        } else {
          addPipelineLog('Image generation failed', 'error');
          reject(new Error('SSE connection failed'));
        }
      };
    });
  };

  // Search for reference images using Freepik API
  const runImageSearch = async (query?: string): Promise<Array<{id: number; url: string; thumbnail: string | null; title: string | null}>> => {
    setIsSearchingImages(true);
    const searchTerm = query || shortDescription || 'cinematic scene';
    addPipelineLog(`Searching for: "${searchTerm.substring(0, 50)}${searchTerm.length > 50 ? '...' : ''}"`, 'info');

    try {
      console.log('Fetching /api/search-images with query:', query);
      const response = await fetch('/api/search-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchTerm }),
      });
      console.log('Response status:', response.status);

      if (!response.ok) {
        const errText = await response.text();
        console.error('API error response text:', errText);
        let errDetail = `API error: ${response.status}`;
        try {
          const errJson = JSON.parse(errText);
          errDetail = errJson.detail || errDetail;
        } catch {
          errDetail = errText || errDetail;
        }
        throw new Error(errDetail);
      }

      const data = await response.json();
      console.log('API response data:', data);
      const images = data.images || [];

      addPipelineLog(`Found ${images.length} reference images`, 'success');
      setReferenceImages(images);

      return images;
    } catch (error) {
      let errorMsg = 'Unknown error';
      if (error instanceof Error) {
        errorMsg = error.message;
      } else if (typeof error === 'string') {
        errorMsg = error;
      } else if (error && typeof error === 'object') {
        errorMsg = JSON.stringify(error);
      }
      console.error('Image search error:', error);
      addPipelineLog(`Image search failed: ${errorMsg}`, 'error');
      throw error;
    } finally {
      setIsSearchingImages(false);
    }
  };

  // Handle Generate button - generates images using available context
  // Mode 1: Has input, no elements → Use raw input as image prompt
  // Mode 2: Has elements, no input, no summary → Stringify elements as prompt
  // Mode 3: Has elements + input → Refine first, then generate images
  // Mode 4: Has elements, no input, has summary → Use summary as prompt
  const handleSubmit = async () => {
    const hasSemanticInput = input.trim().length > 0 || attachments.length > 0;
    const hasElements = sceneData.elements.elements.length > 0;

    // Need some context to generate
    if (!hasSemanticInput && !hasElements) {
      return;
    }

    // Hide references sidebar
    setShowReferencesSidebar(false);

    // Reset cancellation flag
    pipelineCancelledRef.current = false;
    setIsGenerating(true);

    try {
      // MODE 3: Has elements + input → Refine only (no auto image generation)
      if (hasElements && hasSemanticInput) {
        const instruction = input.trim() || 'Analyze and describe this reference as a cinematic scene';
        const currentAttachments = [...attachments];

        // Add user message
        const userMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'user',
          content: input.trim() || `[${currentAttachments.length} attachment(s)]`,
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);

        // Clear input only (keep attachments for potential image generation)
        setInput('');

        // Refine scene (no automatic image generation)
        const { newSceneState } = await runRefinePipeline(instruction, currentAttachments);

        const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Scene refined based on your instructions.',
          timestamp: Date.now(),
          structuredData: newSceneState
        };
        setMessages(prev => [...prev, aiMsg]);

      // MODE 4: Has elements + no input + has summary → Use summary
      } else if (hasElements && !hasSemanticInput && shortDescription) {
        const userMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'user',
          content: '[Generating images from scene narrative]',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);

        await runImageGeneration();

        const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Images generated from scene narrative.',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, aiMsg]);

      // MODE 2: Has elements + no input + no summary → Stringify elements
      } else if (hasElements && !hasSemanticInput) {
        const userMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'user',
          content: '[Generating images from scene structure]',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);

        // Fetch assembled scene and use as prompt
        const response = await fetch('/api/scene/assembled', { cache: 'no-store' });
        if (response.ok) {
          const assembledScene = await response.json();
          const scenePrompt = JSON.stringify(assembledScene, null, 2);

          // Generate images with stringified scene as prompt
          await fetch('/api/generate-images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: scenePrompt, num_images: 4 }),
          });
          await runImageGeneration();
        }

        const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Images generated from scene structure.',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, aiMsg]);

      // MODE 1: Has input, no elements → Use raw input as prompt
      } else if (hasSemanticInput && !hasElements) {
        const rawPrompt = input.trim() || 'cinematic scene';
        const currentAttachments = [...attachments];

        const userMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'user',
          content: input.trim() || `[${currentAttachments.length} attachment(s)]`,
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);

        // Clear input only (keep attachments)
        setInput('');

        // Generate images directly with raw input
        setIsImageGenerating(true);
        setGeneratedImages([]);
        addPipelineLog('Generating images with raw prompt...', 'info');
        addPipelineLog(`Prompt: "${rawPrompt.substring(0, 100)}${rawPrompt.length > 100 ? '...' : ''}"`, 'info');

        const numImages = 4;
        for (let i = 0; i < numImages; i++) {
          if (pipelineCancelledRef.current) break;
          try {
            const response = await fetch('/api/generate-images', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                prompt: rawPrompt,
                num_images: 1,
                aspect_ratio: '1:1',
                output_format: 'png',
                resolution: '1K',
              }),
            });
            if (response.ok) {
              const data = await response.json();
              if (data.images?.length > 0) {
                setGeneratedImages(prev => [...prev, data.images[0]]);
                addPipelineLog(`Image ${i + 1}/${numImages} generated`, 'success');
              }
            }
          } catch (error) {
            addPipelineLog(`Image ${i + 1} failed`, 'warning');
          }
        }
        setIsImageGenerating(false);

        const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Images generated from prompt.',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, aiMsg]);
      }

      setActiveCategory('assembled');
      setSidebarOpen(true);

      // Mark pipeline as complete
      endPipelineLogging(!pipelineCancelledRef.current);

    } catch (error) {
      console.error('Pipeline error:', error);
      // Extract meaningful error message
      let errorMessage = 'Pipeline failed';
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      } else if (error && typeof error === 'object') {
        errorMessage = (error as any).detail || (error as any).message || JSON.stringify(error);
      }
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${errorMessage}`,
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsGenerating(false);
      pipelineCancelledRef.current = false;
    }
  };

  // Handle Assemble button - triggers DSPy pipeline
  // Two modes:
  // 1. Semantic input (text prompt OR file attachments) → Full generation pipeline (LLM required)
  // 2. Manual only (no input, just UI-filled components) → Just show assembled view (no LLM)
  const handleAssemble = async () => {
    const hasSemanticInput = input.trim().length > 0 || attachments.length > 0;
    const hasElements = sceneData.elements.elements.length > 0;

    // Hide references sidebar when assembling
    setShowReferencesSidebar(false);

    // Need either semantic input or existing elements
    if (!hasSemanticInput && !hasElements) {
      startPipelineLogging();
      addPipelineLog('Please enter a scene description, add attachments, or fill in elements first.', 'error');
      endPipelineLogging(false);
      return;
    }

    setIsGenerating(true);

    try {
      // Determine which pipeline to use based on input combination
      // Mode 1: No elements + has semantic input → Full generation pipeline
      // Mode 2: Has elements + no semantic input → Validate/assemble scene (critic + summary)
      // Mode 3: Has elements + has semantic input → Refinement pipeline

      if (hasElements && hasSemanticInput) {
        // MODE 3: Mixed mode - Use refinement pipeline
        // Existing scene + instruction → classifier determines affected heads
        const instruction = input.trim();
        const currentAttachments = [...attachments];

        // Add user message to chat
        const userContent = instruction || `[${currentAttachments.length} attachment(s)]`;
        const userMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'user',
          content: userContent,
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);

        // Clear input only (keep attachments for potential image generation)
        setInput('');

        const { newSceneState } = await runRefinePipeline(instruction, currentAttachments);

        // Add success message
        const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Scene refined based on your instructions.',
          timestamp: Date.now(),
          structuredData: newSceneState
        };
        setMessages(prev => [...prev, aiMsg]);

      } else if (hasSemanticInput) {
        // MODE 1: Semantic only - Full generation pipeline
        // If only file attachments without text, use a default prompt
        const hasTextInput = input.trim().length > 0;
        const prompt = hasTextInput
          ? input.trim()
          : 'Analyze and describe this reference as a cinematic scene';
        const currentAttachments = [...attachments];

        // Add user message to chat
        const userContent = hasTextInput
          ? prompt
          : `[${currentAttachments.length} attachment(s) - generating scene]`;
        const userMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'user',
          content: userContent,
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);

        // Clear input only (keep attachments for potential image generation)
        setInput('');

        const { newSceneState } = await runGeneratePipeline(prompt, currentAttachments);

        // Add success message
        const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Scene generated. You can now refine it with additional instructions.',
          timestamp: Date.now(),
          structuredData: newSceneState
        };
        setMessages(prev => [...prev, aiMsg]);

      } else {
        // MODE 2: Manual only - Validate/assemble scene (critic + summary only)
        const userMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'user',
          content: '[Validating manually configured scene]',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, userMsg]);

        startPipelineLogging();
        addPipelineLog('Validating scene with critic...', 'info');

        const response = await fetch('/api/assemble', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          cache: 'no-store',
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: 'Assembly failed' }));
          throw new Error(err.detail || 'Failed to assemble scene');
        }

        const result = await response.json();

        // Update state with results
        setShortDescription(result.short_description || '');
        setCriticScore(result.critic_score || 0);
        setCriticIssues(result.critic_issues || []);

        addPipelineLog(`Critic score: ${result.critic_score?.toFixed(2) || 'N/A'}`,
          result.critic_score >= 0.85 ? 'success' : 'warning');
        if (result.critic_issues?.length > 0) {
          addPipelineLog('Issues found:', 'warning', result.critic_issues);
        }
        addPipelineLog('Scene validated and summary generated.', 'success');
        endPipelineLogging(true);

        // Build scene state from result
        const newSceneState: SceneState = {
          elements: result.scene?.elements || sceneData.elements,
          objects: { objects: result.scene?.objects || sceneData.objects.objects },
          actions: { actions: result.scene?.actions || sceneData.actions.actions },
          cinematography: result.scene?.cinematography || sceneData.cinematography,
        };

        // Add success message
        const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Scene validated. Critic score: ${result.critic_score?.toFixed(2) || 'N/A'}`,
          timestamp: Date.now(),
          structuredData: newSceneState
        };
        setMessages(prev => [...prev, aiMsg]);
      }

      // Enter refinement mode - scene is now configured
      setIsSceneConfigured(true);

      // Open sidebar to assembled view after pipeline completes
      setActiveCategory('assembled');
      setSidebarOpen(true);

    } catch (error) {
      console.error('Pipeline error:', error);
      // Display error to user
      const errorMsg: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0f0f11] text-zinc-100 font-sans selection:bg-purple-500/30">

      {/* Main Content */}
      <div className={`flex flex-col h-full w-full transition-all duration-300 ease-in-out ${sidebarOpen ? 'md:w-2/3' : 'w-full'}`}>

        {/* Header */}
        <header className="px-6 py-4 border-b border-zinc-900/50 flex items-center justify-between">
          <h1 className="text-lg font-bold tracking-tight text-zinc-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            SceneNapse Studio
          </h1>
          <div className="flex items-center gap-3">
            {/* Sync Status Indicator */}
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full transition-colors ${
                syncStatus === 'synced' ? 'bg-green-500' :
                syncStatus === 'syncing' ? 'bg-yellow-500 animate-pulse' :
                'bg-red-500'
              }`}></span>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                {syncStatus === 'synced' ? 'Synced' : syncStatus === 'syncing' ? 'Syncing...' : 'Error'}
              </span>
            </div>
            <div className="text-xs text-zinc-600 font-mono">v0.1.0</div>
            {/* About button */}
            <button
              onClick={() => setShowAboutModal(true)}
              className="p-2 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
              title="About Scenenapse"
            >
              <HelpCircle className="w-5 h-5" />
            </button>
            {/* Hamburger menu to open Assembled Prompt sidebar */}
            <button
              onClick={() => handleOpenSidebar('assembled')}
              className="p-2 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
              title="Open Assembled Prompt"
            >
              <Menu className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <ChatInterface
          messages={messages}
          generatedImages={generatedImages}
          isImageGenerating={isImageGenerating}
          isGenerating={isGenerating}
          selectedGeneratedId={selectedGeneratedId}
          onSelectGenerated={handleSelectGenerated}
          onOpenImageModal={handleOpenImageModal}
        />

        {/* Input Area */}
        <InputArea
          input={input}
          setInput={setInput}
          attachments={attachments}
          setAttachments={setAttachments}
          onSubmit={handleSubmit}
          onOpenSidebar={handleOpenSidebar}
          onAssemble={handleAssemble}
          sceneData={sceneData}
          onClearCategory={handleClearCategory}
          isGenerating={isGenerating}
          sidebarOpen={sidebarOpen}
          isSceneConfigured={isSceneConfigured}
          onSearchImages={async (query: string) => {
            try {
              await runImageSearch(query);
              // Open references sidebar to show results
              setShowReferencesSidebar(true);
            } catch (error) {
              // Error already logged
            }
          }}
          isSearchingImages={isSearchingImages}
          onDeselectReference={handleDeselectReference}
        />
      </div>

      {/* Right Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        category={activeCategory}
        sceneData={sceneData}
        onClose={handleCloseSidebar}
        onUpdate={handleUpdateSceneData}
        shortDescription={shortDescription}
        criticScore={criticScore}
        criticIssues={criticIssues}
        pipelineLogs={pipelineLogs}
        isPipelineActive={isPipelineActive}
        generatedImages={generatedImages}
        onGenerateImages={async () => {
          const images = await runImageGeneration();
          if (images.length > 0) {
            setGeneratedImages(images);
          }
        }}
        isImageGenerating={isImageGenerating}
        referenceImages={referenceImages}
        onSearchImages={async (query?: string) => {
          try {
            await runImageSearch(query);
          } catch (error) {
            // Error is already logged
          }
        }}
        isSearchingImages={isSearchingImages}
        onOpenRefs={() => setShowReferencesSidebar(true)}
      />

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden backdrop-blur-sm"
          onClick={handleCloseSidebar}
        />
      )}

      {/* About Modal - Carousel */}
      {showAboutModal && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center backdrop-blur-sm"
          onClick={() => setShowAboutModal(false)}
        >
          <div
            className="relative max-w-4xl max-h-[90vh] mx-4 flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={() => setShowAboutModal(false)}
              className="absolute -top-10 right-0 p-2 text-white hover:text-zinc-300 transition-colors z-10"
              title="Close"
            >
              <X className="w-6 h-6" />
            </button>

            {/* Carousel container */}
            <div className="relative flex items-center justify-center">
              {/* Left arrow */}
              <button
                onClick={() => setAboutCarouselIndex((prev) => (prev - 1 + aboutImages.length) % aboutImages.length)}
                className="absolute left-0 -translate-x-12 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors z-10"
                title="Previous"
              >
                <ChevronLeft className="w-6 h-6" />
              </button>

              {/* Image */}
              <img
                src={aboutImages[aboutCarouselIndex]}
                alt={`Scenenapse Workflow Infographic ${aboutCarouselIndex + 1}`}
                className="max-w-full max-h-[80vh] rounded-lg shadow-2xl object-contain"
              />

              {/* Right arrow */}
              <button
                onClick={() => setAboutCarouselIndex((prev) => (prev + 1) % aboutImages.length)}
                className="absolute right-0 translate-x-12 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors z-10"
                title="Next"
              >
                <ChevronRight className="w-6 h-6" />
              </button>
            </div>

            {/* Dots indicator */}
            <div className="flex gap-2 mt-4">
              {aboutImages.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => setAboutCarouselIndex(idx)}
                  className={`w-2.5 h-2.5 rounded-full transition-colors ${
                    idx === aboutCarouselIndex ? 'bg-white' : 'bg-white/40 hover:bg-white/60'
                  }`}
                  title={`Slide ${idx + 1}`}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Generated Image Carousel Modal */}
      {imageModalOpen && generatedImages.length > 0 && (
        <div
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center backdrop-blur-sm"
          onClick={handleCloseImageModal}
        >
          <div
            className="relative max-w-5xl max-h-[90vh] mx-4 flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={handleCloseImageModal}
              className="absolute -top-10 right-0 p-2 text-white hover:text-zinc-300 transition-colors z-10"
              title="Close (selects current image)"
            >
              <X className="w-6 h-6" />
            </button>

            {/* Index badge */}
            <div className="absolute -top-10 left-0 flex items-center gap-2">
              <span className="text-sm font-medium text-white">
                Image {imageModalIndex + 1} of {generatedImages.length}
              </span>
              {selectedGeneratedId === imageModalIndex + 1 && (
                <span className="text-xs bg-cyan-500 text-white px-2 py-0.5 rounded-full">
                  Selected
                </span>
              )}
            </div>

            {/* Carousel container */}
            <div className="relative flex items-center justify-center">
              {/* Left arrow */}
              <button
                onClick={() => setImageModalIndex((prev) => (prev - 1 + generatedImages.length) % generatedImages.length)}
                className="absolute left-0 -translate-x-14 p-3 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors z-10"
                title="Previous"
              >
                <ChevronLeft className="w-8 h-8" />
              </button>

              {/* Full-size image */}
              <img
                src={generatedImages[imageModalIndex].url}
                alt={`Generated image ${imageModalIndex + 1}`}
                className="max-w-full max-h-[80vh] rounded-lg shadow-2xl object-contain"
                style={{ minWidth: '512px', minHeight: '512px' }}
              />

              {/* Right arrow */}
              <button
                onClick={() => setImageModalIndex((prev) => (prev + 1) % generatedImages.length)}
                className="absolute right-0 translate-x-14 p-3 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors z-10"
                title="Next"
              >
                <ChevronRight className="w-8 h-8" />
              </button>
            </div>

            {/* Dots indicator with selection state */}
            <div className="flex gap-3 mt-6">
              {generatedImages.map((_, idx) => {
                const isCurrentImage = idx === imageModalIndex;
                const isSelectedImage = selectedGeneratedId === idx + 1;
                return (
                  <button
                    key={idx}
                    onClick={() => setImageModalIndex(idx)}
                    className={`w-3 h-3 rounded-full transition-all ${
                      isCurrentImage
                        ? isSelectedImage
                          ? 'bg-cyan-500 ring-2 ring-cyan-300'
                          : 'bg-white ring-2 ring-white/50'
                        : isSelectedImage
                          ? 'bg-cyan-500/60 hover:bg-cyan-500'
                          : 'bg-white/40 hover:bg-white/60'
                    }`}
                    title={`Image ${idx + 1}${isSelectedImage ? ' (selected)' : ''}`}
                  />
                );
              })}
            </div>

            {/* Hint text */}
            <p className="text-xs text-zinc-400 mt-3">
              Click outside or press close to select image {imageModalIndex + 1}
            </p>
          </div>
        </div>
      )}

      {/* References Sidebar */}
      <ReferencesSidebar
        isOpen={showReferencesSidebar}
        onClose={() => setShowReferencesSidebar(false)}
        referenceImages={referenceImages}
        onSearchImages={async (query?: string) => {
          try {
            await runImageSearch(query);
          } catch (error) {
            // Error is already logged
          }
        }}
        isSearchingImages={isSearchingImages}
        shortDescription={shortDescription}
        selectedRefId={selectedRefId}
        onToggleSelect={handleToggleRefSelect}
      />

      {/* References Sidebar Overlay */}
      {showReferencesSidebar && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden backdrop-blur-sm"
          onClick={() => setShowReferencesSidebar(false)}
        />
      )}
    </div>
  );
}

// ============================================================
// Chat Interface Component
// ============================================================

// Helper to compute diff between two scene states
function computeSceneDiff(prev: SceneState | undefined, curr: SceneState) {
  const changedElements = new Set<string>();
  const changedObjects = new Set<number>();
  const changedActions = new Set<number>();
  const changedCinematography = new Set<string>();

  if (!prev) return { changedElements, changedObjects, changedActions, changedCinematography };

  // Compare elements by element_id
  const prevElements = prev.elements?.elements || [];
  const currElements = curr.elements?.elements || [];
  const prevElMap = new Map(prevElements.map(el => [el.element_id, el]));
  const currElMap = new Map(currElements.map(el => [el.element_id, el]));

  currElements.forEach(el => {
    const prevEl = prevElMap.get(el.element_id);
    if (!prevEl || JSON.stringify(prevEl) !== JSON.stringify(el)) {
      changedElements.add(el.element_id);
    }
  });
  prevElements.forEach(el => {
    if (!currElMap.has(el.element_id)) {
      changedElements.add(el.element_id);
    }
  });

  // Compare objects by index
  const prevObjects = prev.objects?.objects || [];
  const currObjects = curr.objects?.objects || [];
  const maxObjLen = Math.max(prevObjects.length, currObjects.length);
  for (let i = 0; i < maxObjLen; i++) {
    if (JSON.stringify(prevObjects[i]) !== JSON.stringify(currObjects[i])) {
      changedObjects.add(i);
    }
  }

  // Compare actions by index
  const prevActions = prev.actions?.actions || [];
  const currActions = curr.actions?.actions || [];
  const maxActLen = Math.max(prevActions.length, currActions.length);
  for (let i = 0; i < maxActLen; i++) {
    if (JSON.stringify(prevActions[i]) !== JSON.stringify(currActions[i])) {
      changedActions.add(i);
    }
  }

  // Compare cinematography sub-keys
  const prevCin = prev.cinematography;
  const currCin = curr.cinematography;
  if (JSON.stringify(prevCin?.camera) !== JSON.stringify(currCin?.camera)) changedCinematography.add('camera');
  if (JSON.stringify(prevCin?.lighting) !== JSON.stringify(currCin?.lighting)) changedCinematography.add('lighting');
  if (JSON.stringify(prevCin?.composition) !== JSON.stringify(currCin?.composition)) changedCinematography.add('composition');
  if (JSON.stringify(prevCin?.look) !== JSON.stringify(currCin?.look)) changedCinematography.add('look');

  return { changedElements, changedObjects, changedActions, changedCinematography };
}

interface ChatInterfaceProps {
  messages: ChatMessage[];
  generatedImages: Array<{url: string; width: number; height: number; quality_score?: number | null}>;
  isImageGenerating: boolean;
  isGenerating: boolean;
  selectedGeneratedId: number | null;
  onSelectGenerated: (index: number) => void;
  onOpenImageModal: (index: number) => void;
}

function ChatInterface({ messages, generatedImages, isImageGenerating, isGenerating, selectedGeneratedId, onSelectGenerated, onOpenImageModal }: ChatInterfaceProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, generatedImages, isGenerating]);

  // Find previous assistant message with structuredData for diff computation
  const getPrevStructuredData = (currentIdx: number): SceneState | undefined => {
    for (let i = currentIdx - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant' && messages[i].structuredData) {
        return messages[i].structuredData;
      }
    }
    return undefined;
  };

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar px-4 py-8 space-y-8 scroll-smooth">
      {messages.length === 0 && (
        <div className="h-full flex flex-col items-center justify-center text-zinc-600 space-y-4 opacity-50">
          <Film size={64} strokeWidth={1} />
          <p className="text-lg font-light tracking-wide">Start composing your shot...</p>
        </div>
      )}

      {messages.map((msg, msgIdx) => {
        // Compute diff for this message if it has structuredData
        const prevData = msg.structuredData ? getPrevStructuredData(msgIdx) : undefined;
        const diff = msg.structuredData ? computeSceneDiff(prevData, msg.structuredData) : null;

        return (
        <div key={msg.id} className={`flex gap-4 max-w-4xl mx-auto ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

          {msg.role === 'assistant' && (
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-1 shadow-lg">
              <Aperture size={16} className="text-white" />
            </div>
          )}

          <div className={`flex flex-col gap-2 max-w-[80%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
            <div
              className={`px-5 py-3.5 rounded-2xl leading-relaxed whitespace-pre-wrap shadow-sm ${
                msg.role === 'user'
                  ? 'bg-zinc-800 text-zinc-100 rounded-tr-sm'
                  : 'bg-[#18181b] border border-zinc-800/50 text-zinc-300 rounded-tl-sm shadow-md'
              }`}
            >
              {msg.content}
            </div>

            {/* Structured Data Visualization */}
            {msg.structuredData && (
              <div className="w-full bg-[#131315] border border-zinc-800 rounded-xl overflow-hidden mt-2 shadow-xl">
                <div className="bg-zinc-900/50 px-4 py-2 border-b border-zinc-800 flex justify-between items-center">
                  <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Scene Generated</span>
                  <span className="text-[10px] bg-green-900/30 text-green-400 px-2 py-0.5 rounded-full border border-green-800/50">Ready</span>
                </div>
                <div className="p-3 grid grid-cols-2 gap-2 text-[10px]">
                  {/* Elements Head */}
                  <div className={`bg-zinc-900/30 rounded-lg p-2 border ${diff && diff.changedElements.size > 0 ? 'border-amber-500/50' : 'border-zinc-800/50'}`}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className={`font-semibold uppercase tracking-wide ${diff && diff.changedElements.size > 0 ? 'text-amber-400' : 'text-teal-400'}`}>Elements</span>
                      <span className={diff && diff.changedElements.size > 0 ? 'text-amber-400' : 'text-zinc-500'}>{msg.structuredData.elements?.elements?.length || 0}</span>
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-zinc-400">
                      {msg.structuredData.elements?.elements?.slice(0, 4).map((el, i) => {
                        const isChanged = diff && diff.changedElements.has(el.element_id);
                        return (
                          <span key={el.element_id || i} className={`flex items-center gap-1 ${isChanged ? 'bg-amber-400/20 px-1 rounded' : ''}`}>
                            <span className={el.role ? (isChanged ? 'text-amber-400' : 'text-green-400') : 'text-zinc-600'}>{el.role ? '✓' : '–'}</span>
                            <span className={isChanged ? 'text-amber-400' : 'text-zinc-300'}>{el.role || 'role'}</span>
                          </span>
                        );
                      })}
                      {(!msg.structuredData.elements?.elements?.length) && <span className="text-zinc-600">–</span>}
                    </div>
                  </div>

                  {/* Objects Head */}
                  <div className={`bg-zinc-900/30 rounded-lg p-2 border ${diff && diff.changedObjects.size > 0 ? 'border-amber-500/50' : 'border-zinc-800/50'}`}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className={`font-semibold uppercase tracking-wide ${diff && diff.changedObjects.size > 0 ? 'text-amber-400' : 'text-blue-400'}`}>Objects</span>
                      <span className={diff && diff.changedObjects.size > 0 ? 'text-amber-400' : 'text-zinc-500'}>{msg.structuredData.objects?.objects?.length || 0}</span>
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-zinc-400">
                      {msg.structuredData.objects?.objects?.slice(0, 3).map((obj, i) => {
                        const isChanged = diff && diff.changedObjects.has(i);
                        return (
                        <span key={i} className={`flex items-center gap-1 ${isChanged ? 'bg-amber-400/20 px-1 rounded' : ''}`}>
                          <span className={obj.category ? (isChanged ? 'text-amber-400' : 'text-green-400') : 'text-zinc-600'}>{obj.category ? '✓' : '–'}</span>
                          <span className={isChanged ? 'text-amber-400' : 'text-zinc-300'}>{obj.category || 'object'}</span>
                        </span>
                      )})}
                      {(!msg.structuredData.objects?.objects?.length) && <span className="text-zinc-600">–</span>}
                    </div>
                  </div>

                  {/* Actions Head */}
                  <div className={`bg-zinc-900/30 rounded-lg p-2 border ${diff && diff.changedActions.size > 0 ? 'border-amber-500/50' : 'border-zinc-800/50'}`}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className={`font-semibold uppercase tracking-wide ${diff && diff.changedActions.size > 0 ? 'text-amber-400' : 'text-amber-400'}`}>Actions</span>
                      <span className={diff && diff.changedActions.size > 0 ? 'text-amber-400' : 'text-zinc-500'}>{msg.structuredData.actions?.actions?.length || 0}</span>
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-zinc-400">
                      {msg.structuredData.actions?.actions?.slice(0, 3).map((act, i) => {
                        const isChanged = diff && diff.changedActions.has(i);
                        return (
                        <span key={i} className={`flex items-center gap-1 ${isChanged ? 'bg-amber-400/20 px-1 rounded' : ''}`}>
                          <span className={act.action_class ? (isChanged ? 'text-amber-400' : 'text-green-400') : 'text-zinc-600'}>{act.action_class ? '✓' : '–'}</span>
                          <span className={isChanged ? 'text-amber-400' : 'text-zinc-300'}>{act.action_class || 'action'}</span>
                        </span>
                      )})}
                      {(!msg.structuredData.actions?.actions?.length) && <span className="text-zinc-600">–</span>}
                    </div>
                  </div>

                  {/* Cinematography Head */}
                  <div className={`bg-zinc-900/30 rounded-lg p-2 border ${diff && diff.changedCinematography.size > 0 ? 'border-amber-500/50' : 'border-zinc-800/50'}`}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className={`font-semibold uppercase tracking-wide ${diff && diff.changedCinematography.size > 0 ? 'text-amber-400' : 'text-purple-400'}`}>Cinematography</span>
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-zinc-400">
                      <span className={`flex items-center gap-1 ${diff && diff.changedCinematography.has('camera') ? 'bg-amber-400/20 px-1 rounded' : ''}`}>
                        <span className={msg.structuredData.cinematography?.camera ? (diff && diff.changedCinematography.has('camera') ? 'text-amber-400' : 'text-green-400') : 'text-zinc-600'}>{msg.structuredData.cinematography?.camera ? '✓' : '–'}</span>
                        <span className={diff && diff.changedCinematography.has('camera') ? 'text-amber-400' : 'text-zinc-300'}>Camera</span>
                      </span>
                      <span className={`flex items-center gap-1 ${diff && diff.changedCinematography.has('lighting') ? 'bg-amber-400/20 px-1 rounded' : ''}`}>
                        <span className={msg.structuredData.cinematography?.lighting ? (diff && diff.changedCinematography.has('lighting') ? 'text-amber-400' : 'text-green-400') : 'text-zinc-600'}>{msg.structuredData.cinematography?.lighting ? '✓' : '–'}</span>
                        <span className={diff && diff.changedCinematography.has('lighting') ? 'text-amber-400' : 'text-zinc-300'}>Light</span>
                      </span>
                      <span className={`flex items-center gap-1 ${diff && diff.changedCinematography.has('composition') ? 'bg-amber-400/20 px-1 rounded' : ''}`}>
                        <span className={msg.structuredData.cinematography?.composition ? (diff && diff.changedCinematography.has('composition') ? 'text-amber-400' : 'text-green-400') : 'text-zinc-600'}>{msg.structuredData.cinematography?.composition ? '✓' : '–'}</span>
                        <span className={diff && diff.changedCinematography.has('composition') ? 'text-amber-400' : 'text-zinc-300'}>Comp</span>
                      </span>
                      <span className={`flex items-center gap-1 ${diff && diff.changedCinematography.has('look') ? 'bg-amber-400/20 px-1 rounded' : ''}`}>
                        <span className={msg.structuredData.cinematography?.look ? (diff && diff.changedCinematography.has('look') ? 'text-amber-400' : 'text-green-400') : 'text-zinc-600'}>{msg.structuredData.cinematography?.look ? '✓' : '–'}</span>
                        <span className={diff && diff.changedCinematography.has('look') ? 'text-amber-400' : 'text-zinc-300'}>Look</span>
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {msg.role === 'user' && (
            <div className="w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center shrink-0 mt-1">
              <User size={16} className="text-zinc-300" />
            </div>
          )}
        </div>
      )})}

      {/* Generating Indicator */}
      {isGenerating && (
        <div className="flex gap-4 max-w-4xl mx-auto justify-start">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-1 shadow-lg animate-pulse">
            <Aperture size={16} className="text-white" />
          </div>
          <div className="flex flex-col gap-2 items-start">
            <div className="px-5 py-3.5 rounded-2xl rounded-tl-sm bg-[#18181b] border border-zinc-800/50 shadow-md">
              <div className="flex items-center gap-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-zinc-400 text-sm">Generating scene, please wait...</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Generated Images Section */}
      {(generatedImages.length > 0 || isImageGenerating) && (
        <div className="max-w-4xl mx-auto">
          <div className="bg-[#131315] border border-zinc-800 rounded-xl overflow-hidden shadow-xl">
            <div className="bg-zinc-900/50 px-4 py-2 border-b border-zinc-800 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Sparkles size={14} className="text-purple-400" />
                <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Generated Images</span>
              </div>
              <span className="text-[10px] bg-purple-900/30 text-purple-400 px-2 py-0.5 rounded-full border border-purple-800/50">
                {isImageGenerating
                  ? `${generatedImages.length}/4 generating...`
                  : `${generatedImages.length} image${generatedImages.length !== 1 ? 's' : ''}`
                }
              </span>
            </div>
            <div className="p-3">
              <div className="grid grid-cols-4 gap-2">
                {/* Render generated images as 512x512 thumbnails */}
                {generatedImages.map((img, idx) => {
                  const imageIndex = idx + 1;
                  const isSelected = selectedGeneratedId === imageIndex;
                  return (
                    <button
                      key={`img-${idx}`}
                      onClick={() => onOpenImageModal(idx)}
                      className="block group text-left"
                    >
                      <div className={`relative aspect-square bg-zinc-900 rounded-lg overflow-hidden transition-all ${
                        isSelected
                          ? 'border-2 border-cyan-500 ring-2 ring-cyan-500/30'
                          : 'border border-zinc-800 hover:border-purple-600/50'
                      }`}>
                        <img
                          src={img.url}
                          alt={`Generated image ${imageIndex}`}
                          className="w-full h-full object-cover"
                          style={{ maxWidth: '512px', maxHeight: '512px' }}
                        />
                        <div className="absolute inset-0 bg-purple-500/0 group-hover:bg-purple-500/10 transition-colors" />
                        {/* Index badge */}
                        <div className={`absolute top-1.5 left-1.5 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shadow ${
                          isSelected
                            ? 'bg-cyan-500 text-white'
                            : 'bg-black/70 text-zinc-300'
                        }`}>
                          {imageIndex}
                        </div>
                        {/* Selection checkmark */}
                        {isSelected && (
                          <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-cyan-500 flex items-center justify-center">
                            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        )}
                        {/* Quality score badge */}
                        {img.quality_score != null && (
                          <div className={`absolute bottom-1 left-1 text-[9px] px-1.5 py-0.5 rounded font-medium ${
                            img.quality_score >= 0.7 ? 'bg-green-600/80 text-white' :
                            img.quality_score >= 0.5 ? 'bg-yellow-600/80 text-white' :
                            'bg-red-600/80 text-white'
                          }`}>
                            {(img.quality_score * 100).toFixed(0)}%
                          </div>
                        )}
                        {/* Size badge */}
                        <div className="absolute bottom-1 right-1 bg-black/60 text-[9px] text-zinc-300 px-1.5 py-0.5 rounded">
                          {img.width}×{img.height}
                        </div>
                      </div>
                    </button>
                  );
                })}
                {/* Render loading placeholders for remaining slots */}
                {isImageGenerating && Array.from({ length: 4 - generatedImages.length }).map((_, idx) => (
                  <div
                    key={`placeholder-${idx}`}
                    className="relative aspect-square bg-zinc-900 rounded-lg overflow-hidden border border-zinc-800 border-dashed"
                  >
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <div className="w-6 h-6 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mb-2" />
                      <span className="text-[10px] text-zinc-500">Generating...</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

// ============================================================
// Input Area Component
// ============================================================

interface InputAreaProps {
  input: string;
  setInput: (v: string) => void;
  attachments: Attachment[];
  setAttachments: React.Dispatch<React.SetStateAction<Attachment[]>>;
  onSubmit: () => void;
  onOpenSidebar: (category: CategoryType) => void;
  onAssemble: () => void;
  sceneData: SceneState;
  onClearCategory: (category: CategoryType) => void;
  isGenerating: boolean;
  sidebarOpen: boolean;
  isSceneConfigured: boolean;
  onSearchImages: (query: string) => Promise<void>;
  isSearchingImages: boolean;
  onDeselectReference: () => Promise<void>;
}

function InputArea({
  input, setInput, attachments, setAttachments, onSubmit, onOpenSidebar, onAssemble, sceneData, onClearCategory, isGenerating, sidebarOpen, isSceneConfigured, onSearchImages, isSearchingImages, onDeselectReference
}: InputAreaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showRefWindow, setShowRefWindow] = useState(false);
  const refWindowRef = useRef<HTMLDivElement>(null);
  const refBtnRef = useRef<HTMLButtonElement>(null);
  const [activeFileType, setActiveFileType] = useState<'text' | 'image' | 'video' | 'audio' | null>(null);
  const [showSearchBar, setShowSearchBar] = useState(false);
  const [refSearchQuery, setRefSearchQuery] = useState('');

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  // Click outside listener
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        refWindowRef.current &&
        !refWindowRef.current.contains(event.target as Node) &&
        refBtnRef.current &&
        !refBtnRef.current.contains(event.target as Node)
      ) {
        setShowRefWindow(false);
      }
    };

    if (showRefWindow) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showRefWindow]);

  const hasElements = sceneData.elements.elements.length > 0;
  const hasObjects = sceneData.objects.objects.length > 0;
  const hasActions = sceneData.actions.actions.length > 0;
  const hasCinematography = sceneData.cinematography !== null;
  const hasSemanticInput = input.trim().length > 0 || attachments.length > 0;

  // Generate is disabled when: no input AND no attachments AND no elements, OR already generating
  const canGenerate = (hasSemanticInput || hasElements) && !isGenerating;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canGenerate) {
        onSubmit();
      }
    }
  };

  const [videoNotSupported, setVideoNotSupported] = useState(false);

  const handleFileClick = (type: 'text' | 'image' | 'video' | 'audio') => {
    // Video is not yet supported by DSPy
    if (type === 'video') {
      setVideoNotSupported(true);
      setTimeout(() => setVideoNotSupported(false), 3000);
      return;
    }

    setActiveFileType(type);
    if (fileInputRef.current) {
      switch (type) {
        case 'image': fileInputRef.current.accept = 'image/*'; break;
        case 'audio': fileInputRef.current.accept = 'audio/*'; break;
        case 'text': fileInputRef.current.accept = 'text/plain'; break;
      }
      fileInputRef.current.click();
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0] && activeFileType) {
      const file = e.target.files[0];
      const newAttachment: Attachment = {
        id: Date.now().toString(),
        type: activeFileType,
        name: file.name,
        file: file,
        mimeType: file.type
      };
      setAttachments(prev => [...prev, newAttachment]);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
    setActiveFileType(null);
  };

  const handleRemoveAttachment = async (id: string) => {
    // If removing a reference attachment, call backend to deselect
    if (id.startsWith('ref-')) {
      await onDeselectReference();
    }
    setAttachments(prev => prev.filter(a => a.id !== id));
  };

  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!refSearchQuery.trim() || isSearchingImages) return;

    try {
      await onSearchImages(refSearchQuery.trim());
      // Keep the query visible after search
    } catch (error) {
      // Error is logged by the parent
    }
  };

  const isCreateDisabled = !canGenerate;

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-8 sticky bottom-0 z-20">
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Control Bar */}
      <div className="flex items-center gap-2 mb-3 relative">

        {/* Reference Button */}
        <div className="relative shrink-0">
          <button
            ref={refBtnRef}
            onClick={() => setShowRefWindow(!showRefWindow)}
            className={`flex items-center justify-center rounded-lg border transition-all z-10 relative h-[28px] w-[28px]
              ${showRefWindow || attachments.length > 0
                ? 'bg-zinc-100 text-black border-zinc-100 shadow-[0_0_10px_rgba(255,255,255,0.15)]'
                : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500'}`}
            title="References"
          >
            {attachments.length > 0 ? (
              <span className="text-[10px] font-bold">{attachments.length}</span>
            ) : (
              <Paperclip size={14} />
            )}
          </button>

          {/* Reference Window Popover */}
          {showRefWindow && (
            <div
              ref={refWindowRef}
              className="absolute bottom-10 left-0 w-64 bg-[#18181b] border border-zinc-700 rounded-xl shadow-2xl z-50 overflow-hidden flex flex-col animate-fadeIn"
            >
              <div className="px-3 py-2 border-b border-zinc-800 bg-zinc-900/50 flex justify-between items-center">
                <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">References</span>
                {attachments.length > 0 && (
                  <button
                    onClick={() => setAttachments([])}
                    className="text-[10px] text-red-400 hover:text-red-300 transition-colors"
                  >
                    Clear All
                  </button>
                )}
              </div>

              <div className="max-h-48 overflow-y-auto overflow-x-hidden p-2 space-y-1">
                {attachments.length === 0 && !showSearchBar && (
                  <div className="text-center py-4">
                    <p className="text-zinc-600 text-[10px] italic">No files attached</p>
                  </div>
                )}
                {attachments.map(att => (
                  <div key={att.id} className="flex items-center gap-2 bg-zinc-800/40 hover:bg-zinc-800 p-2 rounded border border-zinc-800/50 transition-colors group">
                    <div className="shrink-0 text-zinc-400">
                      {att.type === 'text' && <FileText size={12} className="text-blue-400" />}
                      {att.type === 'image' && <ImageIcon size={12} className="text-purple-400" />}
                      {att.type === 'video' && <Video size={12} className="text-red-400" />}
                      {att.type === 'audio' && <Mic size={12} className="text-amber-400" />}
                    </div>
                    <span className="text-[11px] text-zinc-300 truncate flex-1 font-medium">{att.name}</span>
                    <button
                      onClick={() => handleRemoveAttachment(att.id)}
                      className="text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all p-0.5"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}

                {/* Search Bar */}
                {showSearchBar && (
                  <form onSubmit={handleSearchSubmit} className="mt-2 overflow-hidden">
                    <div className={`flex items-center gap-2 bg-zinc-800/40 p-2 rounded border ${isSearchingImages ? 'border-cyan-600/50' : 'border-zinc-800/50'} transition-colors overflow-hidden`}>
                      {isSearchingImages ? (
                        <div className="w-3 h-3 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin shrink-0" />
                      ) : (
                        <Search size={12} className="text-cyan-400 shrink-0" />
                      )}
                      <input
                        type="text"
                        value={refSearchQuery}
                        onChange={(e) => setRefSearchQuery(e.target.value)}
                        placeholder="Search Freepik..."
                        disabled={isSearchingImages}
                        className="flex-1 min-w-0 bg-transparent text-[11px] text-zinc-300 placeholder-zinc-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-70"
                      />
                      {refSearchQuery && !isSearchingImages && (
                        <button
                          type="button"
                          onClick={() => setRefSearchQuery('')}
                          className="text-zinc-600 hover:text-zinc-400 p-0.5 shrink-0"
                        >
                          <X size={10} />
                        </button>
                      )}
                      <button
                        type="submit"
                        disabled={!refSearchQuery.trim() || isSearchingImages}
                        className={`text-[10px] px-2 py-0.5 rounded transition-colors shrink-0 whitespace-nowrap ${
                          refSearchQuery.trim() && !isSearchingImages
                            ? 'bg-cyan-900/50 text-cyan-300 hover:bg-cyan-900/70'
                            : 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                        }`}
                      >
                        Search
                      </button>
                    </div>
                  </form>
                )}
              </div>

              {/* Video not supported message */}
              {videoNotSupported && (
                <div className="px-3 py-2 bg-red-900/20 border-t border-red-800/30">
                  <p className="text-[10px] text-red-400 text-center">Video input not yet supported</p>
                </div>
              )}

              {/* Add Buttons */}
              <div className="p-2 border-t border-zinc-800 grid grid-cols-5 gap-1 bg-zinc-900/30">
                <RefAddButton icon={<FileText size={14} />} onClick={() => handleFileClick('text')} color="blue" tooltip="Text" />
                <RefAddButton icon={<ImageIcon size={14} />} onClick={() => handleFileClick('image')} color="purple" tooltip="Image" />
                <RefAddButton icon={<Video size={14} />} onClick={() => handleFileClick('video')} color="red" tooltip="Video (Coming Soon)" disabled />
                <RefAddButton icon={<Mic size={14} />} onClick={() => handleFileClick('audio')} color="amber" tooltip="Audio" />
                <RefAddButton icon={<Search size={14} />} onClick={() => setShowSearchBar(!showSearchBar)} color="cyan" tooltip="Search" active={showSearchBar} />
              </div>
            </div>
          )}
        </div>

        {/* Category Pills */}
        <div className="flex-1 overflow-x-auto no-scrollbar flex items-center gap-2 px-1">
          <CategoryPill
            label="Elements"
            icon={<Layers size={14} />}
            isActive={hasElements}
            onClick={() => onOpenSidebar('elements')}
            onClear={(e) => { e.stopPropagation(); onClearCategory('elements'); }}
            disabled={false}
            color="teal"
            sidebarOpen={sidebarOpen}
          />
          <CategoryPill
            label="Objects"
            icon={<Box size={14} />}
            isActive={hasObjects}
            onClick={() => onOpenSidebar('objects')}
            onClear={(e) => { e.stopPropagation(); onClearCategory('objects'); }}
            disabled={!hasElements}
            color="blue"
            sidebarOpen={sidebarOpen}
          />
          <CategoryPill
            label="Actions"
            icon={<Activity size={14} />}
            isActive={hasActions}
            onClick={() => onOpenSidebar('actions')}
            onClear={(e) => { e.stopPropagation(); onClearCategory('actions'); }}
            disabled={!hasElements}
            color="amber"
            sidebarOpen={sidebarOpen}
          />
          <CategoryPill
            label="Cinematography"
            icon={<Camera size={14} />}
            isActive={hasCinematography}
            onClick={() => onOpenSidebar('cinematography')}
            onClear={(e) => { e.stopPropagation(); onClearCategory('cinematography'); }}
            disabled={!hasElements}
            color="purple"
            sidebarOpen={sidebarOpen}
          />
        </div>

        {/* Assemble Button - triggers DSPy pipeline */}
        {/* Enabled when: has elements OR has semantic input (text/attachments) */}
        <button
          onClick={onAssemble}
          disabled={(!hasElements && !hasSemanticInput) || isGenerating}
          className={`flex-shrink-0 flex items-center justify-center gap-2 h-[28px] w-32 mr-3 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all
            ${((!hasElements && !hasSemanticInput) || isGenerating)
              ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed opacity-50'
              : 'bg-zinc-100 text-black hover:bg-white shadow-[0_0_10px_rgba(255,255,255,0.1)]'}`}
        >
          {isGenerating ? (
            <div className="w-3 h-3 border-2 border-zinc-500 border-t-zinc-800 rounded-full animate-spin" />
          ) : (
            <Layers size={12} />
          )}
          <span>{isGenerating ? 'Running...' : (isSceneConfigured ? 'Refine' : 'Assemble')}</span>
        </button>
      </div>

      {/* Main Input Bar */}
      <div className="relative flex items-end gap-3 bg-[#18181b] p-3 rounded-2xl border border-zinc-800 shadow-2xl ring-1 ring-black/5">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={attachments.length > 0 ? "Describe how these references should influence the scene..." : "Describe your scene..."}
          className="flex-1 bg-transparent text-zinc-200 placeholder-zinc-500 text-base py-2.5 max-h-[200px] resize-none focus:outline-none"
          rows={1}
        />

        <button
          onClick={onSubmit}
          disabled={isCreateDisabled}
          className={`flex-shrink-0 h-10 w-32 flex items-center justify-center gap-2 rounded-lg font-medium transition-all ${
            !isCreateDisabled
              ? isSceneConfigured
                ? 'bg-purple-600 text-white hover:bg-purple-500 shadow-[0_0_15px_rgba(147,51,234,0.3)]'
                : 'bg-zinc-100 text-black hover:bg-white shadow-[0_0_15px_rgba(255,255,255,0.15)]'
              : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
          }`}
        >
          {isGenerating ? (
            <div className="w-5 h-5 border-2 border-zinc-500 border-t-zinc-800 rounded-full animate-spin" />
          ) : (
            <>
              <span>Generate</span>
              <Wand2 size={18} />
            </>
          )}
        </button>
      </div>
      <p className="text-center text-xs text-zinc-600 mt-3">
        AI-generated content may be inaccurate.
      </p>
    </div>
  );
}

// ============================================================
// Category Pill Component
// ============================================================

interface PillProps {
  label: string;
  icon: React.ReactNode;
  isActive: boolean;
  onClick: () => void;
  onClear: (e: React.MouseEvent) => void;
  disabled?: boolean;
  color?: 'teal' | 'blue' | 'purple' | 'amber';
  sidebarOpen: boolean;
}

function CategoryPill({ label, icon, isActive, onClick, onClear, disabled, color = 'blue', sidebarOpen }: PillProps) {
  const colorStyles = {
    teal: 'bg-teal-500/10 border-teal-500/50 text-teal-400 hover:bg-teal-500/20',
    blue: 'bg-blue-500/10 border-blue-500/50 text-blue-400 hover:bg-blue-500/20',
    purple: 'bg-purple-500/10 border-purple-500/50 text-purple-400 hover:bg-purple-500/20',
    amber: 'bg-amber-500/10 border-amber-500/50 text-amber-400 hover:bg-amber-500/20',
  };

  const activeStyle = colorStyles[color];
  const expansionClass = sidebarOpen
    ? 'xl:w-auto xl:px-4 xl:rounded-full'
    : 'md:w-auto md:px-4 md:rounded-full';
  const textDisplayClass = sidebarOpen ? 'hidden xl:inline' : 'hidden md:inline';
  const closeDisplayClass = sidebarOpen ? 'hidden xl:flex' : 'hidden md:flex';

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`group relative flex items-center justify-center gap-2 transition-all border whitespace-nowrap h-[28px]
        w-[28px] p-0 rounded-lg ${expansionClass}
        ${disabled ? 'opacity-40 cursor-not-allowed border-zinc-800 bg-zinc-900 text-zinc-600' : ''}
        ${!disabled && isActive ? activeStyle : !disabled && 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300'}`}
      title={label}
    >
      <span>{icon}</span>
      <span className={`${textDisplayClass} text-xs font-semibold uppercase tracking-wider`}>{label}</span>
      {isActive && (
        <div
          role="button"
          onClick={onClear}
          className={`${closeDisplayClass} ml-1 p-0.5 rounded-full hover:bg-black/20 transition-colors`}
        >
          <X size={12} />
        </div>
      )}
    </button>
  );
}

// ============================================================
// Reference Add Button Component
// ============================================================

function RefAddButton({ icon, onClick, color, tooltip, active, disabled }: { icon: React.ReactNode; onClick: () => void; color: string; tooltip: string; active?: boolean; disabled?: boolean }) {
  const hoverColors: Record<string, string> = {
    blue: 'hover:text-blue-400 hover:bg-blue-500/10 hover:border-blue-500/30',
    purple: 'hover:text-purple-400 hover:bg-purple-500/10 hover:border-purple-500/30',
    red: 'hover:text-red-400 hover:bg-red-500/10 hover:border-red-500/30',
    amber: 'hover:text-amber-400 hover:bg-amber-500/10 hover:border-amber-500/30',
    cyan: 'hover:text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-500/30'
  };
  const activeColors: Record<string, string> = {
    blue: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
    purple: 'text-purple-400 bg-purple-500/10 border-purple-500/30',
    red: 'text-red-400 bg-red-500/10 border-red-500/30',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
    cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30'
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={tooltip}
      className={`p-2 rounded flex justify-center border transition-all ${
        disabled ? 'text-zinc-600 border-transparent cursor-not-allowed opacity-50' :
        active ? activeColors[color] : `text-zinc-500 border-transparent ${hoverColors[color]}`
      }`}
    >
      {icon}
    </button>
  );
}

// ============================================================
// Sidebar Component
// ============================================================

interface PipelineLogEntry {
  id: string;
  timestamp: number;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  issues?: string[];
}

interface SidebarProps {
  isOpen: boolean;
  category: CategoryType;
  sceneData: SceneState;
  onClose: () => void;
  onUpdate: (category: CategoryType, data: any) => void;
  shortDescription: string;
  criticScore: number;
  criticIssues: string[];
  pipelineLogs: PipelineLogEntry[];
  isPipelineActive: boolean;
  generatedImages: Array<{url: string; width: number; height: number; quality_score?: number | null}>;
  onGenerateImages: () => Promise<void>;
  isImageGenerating: boolean;
  referenceImages: Array<{id: number; url: string; thumbnail: string | null; title: string | null}>;
  onSearchImages: (query?: string) => Promise<void>;
  isSearchingImages: boolean;
  onOpenRefs: () => void;
}

function Sidebar({ isOpen, category, sceneData, onClose, onUpdate, shortDescription, criticScore, criticIssues, pipelineLogs, isPipelineActive, generatedImages, onGenerateImages, isImageGenerating, referenceImages, onSearchImages, isSearchingImages, onOpenRefs }: SidebarProps) {
  const [activeTab, setActiveTab] = useState<string>('camera');
  const [showLogsView, setShowLogsView] = useState(false);

  // Reset to assembled view when sidebar closes or category changes
  useEffect(() => {
    if (!isOpen || category !== 'assembled') {
      setShowLogsView(false);
    }
  }, [isOpen, category]);

  // Element UI State
  const [isAddingElement, setIsAddingElement] = useState(false);
  const [editingElementIndex, setEditingElementIndex] = useState<number | null>(null);
  const [newElementForm, setNewElementForm] = useState<Partial<Element>>({
    role: '',
    entity_type: '',
    importance: 'secondary',
    rough_description: ''
  });

  // Object UI State
  const [isAddingObject, setIsAddingObject] = useState(false);
  const [editingObjectIndex, setEditingObjectIndex] = useState<number | null>(null);
  const [newObjectForm, setNewObjectForm] = useState<Partial<SceneObject>>({
    category: 'human',
    description: '',
    dependencies: []
  });

  // Action UI State
  const [isAddingAction, setIsAddingAction] = useState(false);
  const [editingActionIndex, setEditingActionIndex] = useState<number | null>(null);
  const [newActionForm, setNewActionForm] = useState<Partial<Action>>({
    dependencies: [],
    action_class: 'Walking',
    stage_class: 'Ongoing',
    description: '',
    temporal_context: { is_highlight_frame: false, frame_position_in_event: 'ongoing' }
  });

  // Reset states when category changes
  useEffect(() => {
    if (category === 'cinematography') setActiveTab('camera');
    setIsAddingElement(false);
    setIsAddingObject(false);
    setIsAddingAction(false);
    setEditingElementIndex(null);
    setEditingObjectIndex(null);
    setEditingActionIndex(null);
  }, [category]);

  // Cinematography handlers
  const handleCinematographyChange = (section: keyof Cinematography, field: string, value: string) => {
    const currentCinema = sceneData.cinematography || DEFAULT_CINEMATOGRAPHY;
    const currentSection = currentCinema[section] || (DEFAULT_CINEMATOGRAPHY as any)[section];

    const updated = {
      ...currentCinema,
      [section]: {
        ...currentSection,
        [field]: value
      }
    };
    onUpdate('cinematography', updated);
  };

  const handleCinematographyDependencyToggle = (elementId: string) => {
    const currentCinema = sceneData.cinematography || DEFAULT_CINEMATOGRAPHY;
    const currentDeps = currentCinema.dependencies || [];

    let newDeps;
    if (currentDeps.includes(elementId)) {
      newDeps = currentDeps.filter(id => id !== elementId);
    } else {
      newDeps = [...currentDeps, elementId];
    }

    const updated = {
      ...currentCinema,
      dependencies: newDeps
    };
    onUpdate('cinematography', updated);
  };

  // Element handlers
  const handleEditElement = (index: number) => {
    const el = sceneData.elements.elements[index];
    setNewElementForm({ ...el });
    setEditingElementIndex(index);
    setIsAddingElement(true);
  };

  const handleSaveNewElement = () => {
    if (!newElementForm.role || !newElementForm.entity_type) return;

    let elementId = newElementForm.element_id;
    if (!elementId) {
      const currentCount = sceneData.elements.elements.length;
      elementId = `element_${currentCount}`;
    }

    const newElement: Element = {
      element_id: elementId,
      role: newElementForm.role!,
      entity_type: newElementForm.entity_type!,
      importance: newElementForm.importance || 'secondary',
      rough_description: newElementForm.rough_description || ''
    };

    let updatedElements;
    if (editingElementIndex !== null) {
      updatedElements = [...sceneData.elements.elements];
      updatedElements[editingElementIndex] = newElement;
    } else {
      updatedElements = [...sceneData.elements.elements, newElement];
    }

    onUpdate('elements', { elements: updatedElements });
    setNewElementForm({ role: '', entity_type: '', importance: 'secondary', rough_description: '' });
    setEditingElementIndex(null);
    setIsAddingElement(false);
  };

  const handleDeleteElement = (index: number) => {
    const updatedElements = sceneData.elements.elements.filter((_, i) => i !== index);
    onUpdate('elements', { elements: updatedElements });
  };

  // Object handlers
  const handleEditObject = (index: number) => {
    const obj = sceneData.objects.objects[index];
    setNewObjectForm({ ...obj });
    setEditingObjectIndex(index);
    setIsAddingObject(true);
  };

  const handleSaveNewObject = () => {
    if (!newObjectForm.description || !newObjectForm.dependencies?.length) return;

    const newObject: SceneObject = {
      dependencies: newObjectForm.dependencies,
      category: newObjectForm.category || 'human',
      description: newObjectForm.description,
      location: newObjectForm.location || 'center',
      relative_size: newObjectForm.relative_size || 'medium',
      shape_and_color: newObjectForm.shape_and_color || 'undefined',
      texture: newObjectForm.texture || 'undefined',
      appearance_details: newObjectForm.appearance_details || 'undefined'
    };

    let updatedObjects;
    if (editingObjectIndex !== null) {
      updatedObjects = [...sceneData.objects.objects];
      updatedObjects[editingObjectIndex] = newObject;
    } else {
      updatedObjects = [...sceneData.objects.objects, newObject];
    }

    onUpdate('objects', { objects: updatedObjects });
    setNewObjectForm({ category: 'human', description: '', dependencies: [] });
    setEditingObjectIndex(null);
    setIsAddingObject(false);
  };

  const handleDeleteObject = (index: number) => {
    const updatedObjects = sceneData.objects.objects.filter((_, i) => i !== index);
    onUpdate('objects', { objects: updatedObjects });
  };

  // Action handlers
  const handleEditAction = (index: number) => {
    const act = sceneData.actions.actions[index];
    setNewActionForm({ ...act });
    setEditingActionIndex(index);
    setIsAddingAction(true);
  };

  const handleSaveNewAction = () => {
    if (!newActionForm.description || !newActionForm.dependencies?.length) return;

    const newAction: Action = {
      dependencies: newActionForm.dependencies,
      action_class: newActionForm.action_class || 'Walking',
      stage_class: newActionForm.stage_class || 'Ongoing',
      description: newActionForm.description,
      temporal_context: newActionForm.temporal_context || { is_highlight_frame: false, frame_position_in_event: 'ongoing' }
    };

    let updatedActions;
    if (editingActionIndex !== null) {
      updatedActions = [...sceneData.actions.actions];
      updatedActions[editingActionIndex] = newAction;
    } else {
      updatedActions = [...sceneData.actions.actions, newAction];
    }

    onUpdate('actions', { actions: updatedActions });
    setNewActionForm({
      dependencies: [],
      action_class: 'Walking',
      stage_class: 'Ongoing',
      description: '',
      temporal_context: { is_highlight_frame: false, frame_position_in_event: 'ongoing' }
    });
    setEditingActionIndex(null);
    setIsAddingAction(false);
  };

  const handleDeleteAction = (index: number) => {
    const updatedActions = sceneData.actions.actions.filter((_, i) => i !== index);
    onUpdate('actions', { actions: updatedActions });
  };

  if (!isOpen) return null;

  const getCategoryTitle = (cat: CategoryType) => {
    if (cat === 'objects') return 'Object Descriptions';
    if (cat === 'cinematography') return 'Cinematography';
    if (cat === 'assembled') return 'Assembled Prompt';
    if (cat === 'actions') return 'Actions';
    return 'Elements';
  };

  return (
    <div className={`fixed top-0 right-0 h-full w-full md:w-1/3 bg-[#18181b] border-l border-zinc-800 shadow-2xl transform transition-transform duration-300 ease-in-out z-50 ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>

      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-zinc-800 bg-[#18181b]/95 backdrop-blur">
        <div className="flex items-center gap-3">
          {category === 'cinematography' && <Camera className="w-6 h-6 text-purple-400" />}
          {category === 'objects' && <Box className="w-6 h-6 text-blue-400" />}
          {category === 'actions' && <Activity className="w-6 h-6 text-amber-400" />}
          {category === 'elements' && <Layers className="w-6 h-6 text-teal-400" />}
          {category === 'assembled' && (
            showLogsView || isPipelineActive
              ? <Terminal className="w-6 h-6 text-cyan-400" />
              : <FileJson className="w-6 h-6 text-emerald-400" />
          )}
          {category === 'assembled' ? (
            <h2 className="text-xl font-semibold tracking-wide text-zinc-100">
              {showLogsView || isPipelineActive ? 'Pipeline Log' : 'Assembled Prompt'}
            </h2>
          ) : (
            <h2 className="text-xl font-semibold capitalize tracking-wide text-zinc-100">{getCategoryTitle(category)}</h2>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Reference Images button - always visible in assembled view */}
          {category === 'assembled' && (
            <button
              onClick={onOpenRefs}
              className={`p-2 hover:bg-zinc-800 rounded-full transition-colors ${
                referenceImages.length > 0
                  ? 'text-cyan-400 hover:text-cyan-300'
                  : 'text-zinc-400 hover:text-cyan-400'
              }`}
              title={referenceImages.length > 0 ? `Reference Images (${referenceImages.length})` : 'Search Reference Images'}
            >
              <Search className="w-5 h-5" />
            </button>
          )}
          {/* Assembled Prompt button - show when viewing logs (not during active pipeline) */}
          {category === 'assembled' && !isPipelineActive && showLogsView && (
            <button
              onClick={() => setShowLogsView(false)}
              className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-emerald-400 transition-colors"
              title="View Assembled Prompt"
            >
              <FileJson className="w-5 h-5" />
            </button>
          )}
          {/* Logs toggle button - show when viewing assembled prompt */}
          {category === 'assembled' && !isPipelineActive && !showLogsView && (
            <button
              onClick={() => setShowLogsView(true)}
              className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-cyan-400 transition-colors"
              title="View Pipeline Log"
            >
              <Terminal className="w-5 h-5" />
            </button>
          )}
          <button onClick={onClose} className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex h-[calc(100%-80px)]">

        {/* Vertical Tabs for Cinematography */}
        {category === 'cinematography' && (
          <div className="w-16 bg-[#131315] border-r border-zinc-800 flex flex-col items-center py-4 gap-4">
            <TabButton icon={<Camera size={20} />} active={activeTab === 'camera'} onClick={() => setActiveTab('camera')} label="Cam" />
            <TabButton icon={<Zap size={20} />} active={activeTab === 'lighting'} onClick={() => setActiveTab('lighting')} label="Light" />
            <TabButton icon={<Layout size={20} />} active={activeTab === 'composition'} onClick={() => setActiveTab('composition')} label="Comp" />
            <TabButton icon={<Palette size={20} />} active={activeTab === 'look'} onClick={() => setActiveTab('look')} label="Look" />
          </div>
        )}

        {/* Form Content - Assembled has its own scroll control */}
        {category === 'assembled' ? (
          <div className="flex-1 overflow-hidden p-6">
            <AssembledView
              sceneData={sceneData}
              shortDescription={shortDescription}
              criticScore={criticScore}
              criticIssues={criticIssues}
              pipelineLogs={pipelineLogs}
              isPipelineActive={isPipelineActive}
              showLogsView={showLogsView}
            />
          </div>
        ) : (
        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* ELEMENTS */}
          {category === 'elements' && (
            <div className="space-y-6 animate-fadeIn">
              {!isAddingElement ? (
                <>
                  <div className="flex justify-between items-center mb-4">
                    <p className="text-zinc-400 text-sm">Elements define the who and what of your scene.</p>
                    <button
                      onClick={() => {
                        const isFirst = sceneData.elements.elements.length === 0;
                        setNewElementForm({
                          role: '',
                          entity_type: '',
                          importance: isFirst ? 'primary' : 'secondary',
                          rough_description: ''
                        });
                        setEditingElementIndex(null);
                        setIsAddingElement(true);
                      }}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-teal-500/10 text-teal-400 rounded-lg text-xs font-bold uppercase hover:bg-teal-500/20 border border-teal-500/50 transition-all"
                    >
                      <Plus size={14} /> Add Element
                    </button>
                  </div>

                  {sceneData.elements.elements.length === 0 ? (
                    <div className="text-center py-12 border border-dashed border-zinc-800 rounded-xl">
                      <Layers className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
                      <p className="text-zinc-500 text-sm">No elements defined.</p>
                      <p className="text-zinc-600 text-xs mt-1">Add elements to unlock Objects, Camera, and Action.</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {sceneData.elements.elements.map((el, idx) => (
                        <div
                          key={idx}
                          onClick={() => handleEditElement(idx)}
                          className="bg-zinc-900 border border-zinc-800 p-4 rounded-xl hover:border-teal-500/30 cursor-pointer transition-all group relative"
                        >
                          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex gap-2">
                            <button
                              onClick={(e) => { e.stopPropagation(); handleDeleteElement(idx); }}
                              className="p-1.5 bg-red-900/20 text-red-400 rounded hover:bg-red-900/40"
                            >
                              <Trash2 size={12} />
                            </button>
                            <div className="p-1.5 bg-zinc-800 text-zinc-400 rounded">
                              <Edit2 size={12} />
                            </div>
                          </div>
                          <div className="flex justify-between items-start">
                            <div className="flex items-center gap-2">
                              <span className="bg-zinc-800 text-zinc-500 text-[10px] px-1.5 py-0.5 rounded font-mono">#{idx + 1}</span>
                              <span className="font-semibold text-teal-100">{el.role}</span>
                            </div>
                            <span className="text-[10px] uppercase tracking-wider text-zinc-500">{el.importance}</span>
                          </div>
                          <div className="text-xs text-zinc-400 mt-1">
                            <span className="text-zinc-500">Type: </span>{el.entity_type}
                          </div>
                          {el.rough_description && (
                            <div className="text-xs text-zinc-500 italic mt-2 border-l-2 border-zinc-800 pl-2">
                              &quot;{el.rough_description}&quot;
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="animate-fadeIn">
                  <button onClick={() => setIsAddingElement(false)} className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 mb-4">
                    <ArrowLeft size={14} /> Back to list
                  </button>
                  <h3 className="text-lg font-medium text-white mb-6">{editingElementIndex !== null ? 'Edit Element' : 'Add New Element'}</h3>

                  <div className="space-y-4">
                    <InputField
                      label="Role (Required)"
                      info="The narrative role, e.g., 'Protagonist' or 'Background'."
                      value={newElementForm.role ?? ''}
                      onChange={(v) => setNewElementForm(p => ({ ...p, role: v }))}
                      placeholder="e.g. Main Subject, Hero, Villain..."
                    />
                    <InputField
                      label="Entity Type (Required)"
                      info="Semantic description of what this entity is."
                      value={newElementForm.entity_type ?? ''}
                      onChange={(v) => setNewElementForm(p => ({ ...p, entity_type: v }))}
                      placeholder="e.g. Adult woman, Red sports car..."
                    />
                    <SelectField
                      label="Importance"
                      info="Visual dominance in the frame."
                      value={newElementForm.importance ?? 'secondary'}
                      options={['primary', 'secondary', 'background']}
                      onChange={(v) => setNewElementForm(p => ({ ...p, importance: v }))}
                    />
                    <InputField
                      label="Brief Description"
                      info="A brief phrase describing the entity."
                      value={newElementForm.rough_description ?? ''}
                      onChange={(v) => setNewElementForm(p => ({ ...p, rough_description: v }))}
                      placeholder="e.g. A tall woman in a red dress..."
                    />

                    <button
                      onClick={handleSaveNewElement}
                      disabled={!newElementForm.role || !newElementForm.entity_type}
                      className={`w-full py-3 rounded-xl flex items-center justify-center gap-2 font-medium transition-all mt-4 ${
                        !newElementForm.role || !newElementForm.entity_type
                          ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                          : 'bg-teal-600 text-white hover:bg-teal-500 shadow-lg shadow-teal-900/20'
                      }`}
                    >
                      <Save size={18} /> {editingElementIndex !== null ? 'Update Element' : 'Save Element'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* OBJECTS */}
          {category === 'objects' && (
            <div className="space-y-6 animate-fadeIn">
              {!isAddingObject ? (
                <>
                  <div className="flex justify-between items-center mb-4">
                    <p className="text-zinc-400 text-sm">Visual details of elements.</p>
                    <button
                      onClick={() => {
                        const defaultDep = sceneData.elements.elements.length > 0 ? [sceneData.elements.elements[0].element_id] : [];
                        setNewObjectForm({ category: 'human', description: '', dependencies: defaultDep });
                        setEditingObjectIndex(null);
                        setIsAddingObject(true);
                      }}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/10 text-blue-400 rounded-lg text-xs font-bold uppercase hover:bg-blue-500/20 border border-blue-500/50 transition-all"
                    >
                      <Plus size={14} /> Add Object
                    </button>
                  </div>
                  {sceneData.objects.objects.length === 0 ? (
                    <div className="text-center py-8 text-zinc-600 italic">No objects defined yet.</div>
                  ) : (
                    sceneData.objects.objects.map((obj, idx) => (
                      <div
                        key={idx}
                        onClick={() => handleEditObject(idx)}
                        className="mb-4 p-3 bg-zinc-800 rounded border border-zinc-700 hover:border-blue-500/30 cursor-pointer transition-all group relative"
                      >
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex gap-2">
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDeleteObject(idx); }}
                            className="p-1.5 bg-red-900/20 text-red-400 rounded hover:bg-red-900/40"
                          >
                            <Trash2 size={12} />
                          </button>
                          <div className="p-1.5 bg-zinc-700 text-zinc-400 rounded">
                            <Edit2 size={12} />
                          </div>
                        </div>
                        <div className="flex justify-between items-center mb-1">
                          <div className="font-semibold text-blue-300">{obj.category}</div>
                          {obj.dependencies.length > 0 && (
                            <span className="text-[10px] bg-zinc-900 text-zinc-500 px-1.5 py-0.5 rounded">
                              Ref: {sceneData.elements.elements.find(e => e.element_id === obj.dependencies[0])?.role || obj.dependencies[0]}
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-zinc-400 line-clamp-3">{obj.description}</div>
                      </div>
                    ))
                  )}
                </>
              ) : (
                <div className="animate-fadeIn">
                  <button onClick={() => setIsAddingObject(false)} className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 mb-4">
                    <ArrowLeft size={14} /> Back to list
                  </button>
                  <h3 className="text-lg font-medium text-white mb-6">{editingObjectIndex !== null ? 'Edit Object' : 'Add New Object'}</h3>

                  <div className="space-y-4">
                    <div className="flex flex-col gap-1.5">
                      <div className="flex items-center gap-1.5">
                        <label className="text-xs font-medium text-zinc-400">Linked Element (Required)</label>
                        <InfoTooltip text="Connects this visual description to a semantic element." />
                      </div>
                      <select
                        value={newObjectForm.dependencies?.[0] ?? ''}
                        onChange={(e) => setNewObjectForm(p => ({ ...p, dependencies: [e.target.value] }))}
                        className="w-full bg-[#0f0f11] border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-zinc-500"
                      >
                        <option value="">Select an element...</option>
                        {sceneData.elements.elements.map(el => (
                          <option key={el.element_id} value={el.element_id}>{el.role} ({el.entity_type})</option>
                        ))}
                      </select>
                    </div>

                    <InputField
                      label="Category"
                      info="High level category."
                      value={newObjectForm.category ?? ''}
                      onChange={(v) => setNewObjectForm(p => ({ ...p, category: v }))}
                      placeholder="e.g. human, animal, object"
                    />

                    <TextAreaField
                      label="Visual Description (Required)"
                      info="Rich, detailed description of visual appearance."
                      value={newObjectForm.description ?? ''}
                      onChange={(v) => setNewObjectForm(p => ({ ...p, description: v }))}
                      placeholder="e.g. She wears a flowing crimson silk gown..."
                    />

                    <button
                      onClick={handleSaveNewObject}
                      disabled={!newObjectForm.description || !newObjectForm.dependencies?.length}
                      className={`w-full py-3 rounded-xl flex items-center justify-center gap-2 font-medium transition-all mt-4 ${
                        !newObjectForm.description || !newObjectForm.dependencies?.length
                          ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                          : 'bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-900/20'
                      }`}
                    >
                      <Save size={18} /> {editingObjectIndex !== null ? 'Update Object' : 'Save Object'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ACTIONS */}
          {category === 'actions' && (
            <div className="space-y-6 animate-fadeIn">
              {!isAddingAction ? (
                <>
                  <div className="flex justify-between items-center mb-4">
                    <p className="text-zinc-400 text-sm">Motion and temporal dynamics.</p>
                    <button
                      onClick={() => {
                        const defaultDep = sceneData.elements.elements.length > 0 ? [sceneData.elements.elements[0].element_id] : [];
                        setNewActionForm({
                          dependencies: defaultDep,
                          action_class: 'Walking',
                          stage_class: 'Ongoing',
                          description: '',
                          temporal_context: { is_highlight_frame: false, frame_position_in_event: 'ongoing' }
                        });
                        setEditingActionIndex(null);
                        setIsAddingAction(true);
                      }}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/10 text-amber-400 rounded-lg text-xs font-bold uppercase hover:bg-amber-500/20 border border-amber-500/50 transition-all"
                    >
                      <Plus size={14} /> Add Action
                    </button>
                  </div>
                  {sceneData.actions.actions.length === 0 ? (
                    <div className="text-center py-8 text-zinc-600 italic">No actions defined yet.</div>
                  ) : (
                    sceneData.actions.actions.map((act, idx) => (
                      <div
                        key={idx}
                        onClick={() => handleEditAction(idx)}
                        className="mb-4 p-3 bg-zinc-800 rounded border border-zinc-700 hover:border-amber-500/30 cursor-pointer transition-all group relative"
                      >
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex gap-2">
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDeleteAction(idx); }}
                            className="p-1.5 bg-red-900/20 text-red-400 rounded hover:bg-red-900/40"
                          >
                            <Trash2 size={12} />
                          </button>
                          <div className="p-1.5 bg-zinc-700 text-zinc-400 rounded">
                            <Edit2 size={12} />
                          </div>
                        </div>
                        <div className="flex justify-between items-center mb-1">
                          <div className="font-semibold text-amber-300">{act.action_class}</div>
                          {act.dependencies.length > 0 && (
                            <span className="text-[10px] bg-zinc-900 text-zinc-500 px-1.5 py-0.5 rounded">
                              Ref: {sceneData.elements.elements.find(e => e.element_id === act.dependencies[0])?.role || act.dependencies[0]}
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-zinc-400 line-clamp-2">{act.description}</div>
                      </div>
                    ))
                  )}
                </>
              ) : (
                <div className="animate-fadeIn">
                  <button onClick={() => setIsAddingAction(false)} className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 mb-4">
                    <ArrowLeft size={14} /> Back to list
                  </button>
                  <h3 className="text-lg font-medium text-white mb-6">{editingActionIndex !== null ? 'Edit Action' : 'Add New Action'}</h3>

                  <div className="space-y-4">
                    <div className="flex flex-col gap-1.5">
                      <div className="flex items-center gap-1.5">
                        <label className="text-xs font-medium text-zinc-400">Target Element (Required)</label>
                        <InfoTooltip text="The entity performing or experiencing this action." />
                      </div>
                      <select
                        value={newActionForm.dependencies?.[0] ?? ''}
                        onChange={(e) => setNewActionForm(p => ({ ...p, dependencies: [e.target.value] }))}
                        className="w-full bg-[#0f0f11] border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-zinc-500"
                      >
                        <option value="">Select an element...</option>
                        {sceneData.elements.elements.map(el => (
                          <option key={el.element_id} value={el.element_id}>{el.role} ({el.entity_type})</option>
                        ))}
                      </select>
                    </div>

                    <InputField
                      label="Action Type"
                      info="What kind of action is it?"
                      value={newActionForm.action_class ?? ''}
                      onChange={(v) => setNewActionForm(p => ({ ...p, action_class: v }))}
                      placeholder="e.g. Walking, Falling, Smiling"
                    />

                    <InputField
                      label="Stage or Phase"
                      info="Phase of the action."
                      value={newActionForm.stage_class ?? ''}
                      onChange={(v) => setNewActionForm(p => ({ ...p, stage_class: v }))}
                      placeholder="e.g. Ongoing, Peak, Ending"
                    />

                    <TextAreaField
                      label="Description (Required)"
                      info="Vivid description of the motion physics."
                      value={newActionForm.description ?? ''}
                      onChange={(v) => setNewActionForm(p => ({ ...p, description: v }))}
                      placeholder="e.g. Strides gracefully along the path..."
                    />

                    <button
                      onClick={handleSaveNewAction}
                      disabled={!newActionForm.description || !newActionForm.dependencies?.length}
                      className={`w-full py-3 rounded-xl flex items-center justify-center gap-2 font-medium transition-all mt-4 ${
                        !newActionForm.description || !newActionForm.dependencies?.length
                          ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                          : 'bg-amber-600 text-white hover:bg-amber-500 shadow-lg shadow-amber-900/20'
                      }`}
                    >
                      <Save size={18} /> {editingActionIndex !== null ? 'Update Action' : 'Save Action'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* CINEMATOGRAPHY */}
          {category === 'cinematography' && (
            <>
              {activeTab === 'camera' && (
                <div className="space-y-4 animate-fadeIn">
                  <div className="p-4 bg-zinc-900/50 rounded-xl border border-zinc-800 mb-6">
                    <div className="flex items-center gap-2 mb-3">
                      <Target size={14} className="text-purple-400" />
                      <h3 className="text-xs uppercase tracking-wider text-zinc-400 font-bold">Target Elements</h3>
                    </div>
                    {sceneData.elements.elements.length === 0 ? (
                      <p className="text-xs text-zinc-600 italic">No elements available to target.</p>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {sceneData.elements.elements.map(el => {
                          const isSelected = (sceneData.cinematography?.dependencies || []).includes(el.element_id);
                          return (
                            <button
                              key={el.element_id}
                              onClick={() => handleCinematographyDependencyToggle(el.element_id)}
                              className={`text-xs px-2 py-1 rounded border transition-all ${
                                isSelected
                                  ? 'bg-purple-900/30 border-purple-500 text-purple-200'
                                  : 'bg-zinc-800 border-zinc-700 text-zinc-500 hover:bg-zinc-700'
                              }`}
                            >
                              {el.role}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  <h3 className="text-sm uppercase tracking-wider text-zinc-500 font-bold mb-4">Camera Setup</h3>
                  <SelectField
                    label="Shot Size"
                    info="How much of the subject fills the frame."
                    value={sceneData.cinematography?.camera?.shot_size ?? 'medium'}
                    options={SHOT_SIZES}
                    onChange={(v) => handleCinematographyChange('camera', 'shot_size', v)}
                  />
                  <SelectField
                    label="Angle"
                    info="Vertical angle of the camera."
                    value={sceneData.cinematography?.camera?.camera_angle ?? 'eye_level'}
                    options={CAMERA_ANGLES}
                    onChange={(v) => handleCinematographyChange('camera', 'camera_angle', v)}
                  />
                  <InputField
                    label="Lens Size"
                    info="Focal length of the lens."
                    value={sceneData.cinematography?.camera?.lens_size ?? '35mm'}
                    placeholder="e.g. 35mm, 85mm"
                    onChange={(v) => handleCinematographyChange('camera', 'lens_size', v)}
                  />
                  <InputField
                    label="Movement"
                    info="Camera motion during the shot."
                    value={sceneData.cinematography?.camera?.movement ?? 'static'}
                    placeholder="e.g. static, dolly_in"
                    onChange={(v) => handleCinematographyChange('camera', 'movement', v)}
                  />
                </div>
              )}

              {activeTab === 'lighting' && (
                <div className="space-y-4 animate-fadeIn">
                  <h3 className="text-sm uppercase tracking-wider text-zinc-500 font-bold mb-4">Lighting Design</h3>
                  <SelectField
                    label="Conditions"
                    info="Overall lighting environment."
                    value={sceneData.cinematography?.lighting?.conditions ?? 'soft natural daylight'}
                    options={LIGHTING_CONDITIONS}
                    onChange={(v) => handleCinematographyChange('lighting', 'conditions', v)}
                  />
                  <InputField
                    label="Direction"
                    info="Where the main light source comes from."
                    value={sceneData.cinematography?.lighting?.direction ?? 'frontal'}
                    placeholder="e.g. rim_light, side_left"
                    onChange={(v) => handleCinematographyChange('lighting', 'direction', v)}
                  />
                  <InputField
                    label="Mood Tag"
                    info="Emotional quality of the lighting."
                    value={sceneData.cinematography?.lighting?.mood_tag ?? 'neutral'}
                    placeholder="e.g. noir, romantic"
                    onChange={(v) => handleCinematographyChange('lighting', 'mood_tag', v)}
                  />
                </div>
              )}

              {activeTab === 'composition' && (
                <div className="space-y-4 animate-fadeIn">
                  <h3 className="text-sm uppercase tracking-wider text-zinc-500 font-bold mb-4">Framing & Composition</h3>
                  <TextAreaField
                    label="Description"
                    info="How elements are arranged in the frame."
                    value={sceneData.cinematography?.composition?.description ?? ''}
                    placeholder="Describe the visual arrangement..."
                    onChange={(v) => handleCinematographyChange('composition', 'description', v)}
                  />
                  <InputField
                    label="Subject Layout"
                    info="Where the main subject is placed."
                    value={sceneData.cinematography?.composition?.subject_layout ?? 'centered'}
                    placeholder="e.g. rule_of_thirds"
                    onChange={(v) => handleCinematographyChange('composition', 'subject_layout', v)}
                  />
                </div>
              )}

              {activeTab === 'look' && (
                <div className="space-y-4 animate-fadeIn">
                  <h3 className="text-sm uppercase tracking-wider text-zinc-500 font-bold mb-4">Visual Style</h3>
                  <SelectField
                    label="Artistic Style"
                    info="The overall aesthetic or art style."
                    value={sceneData.cinematography?.look?.artistic_style ?? 'photorealistic'}
                    options={ARTISTIC_STYLES}
                    onChange={(v) => handleCinematographyChange('look', 'artistic_style', v)}
                  />
                  <InputField
                    label="Color Scheme"
                    info="Dominant colors or palette."
                    value={sceneData.cinematography?.look?.color_scheme ?? 'natural'}
                    placeholder="e.g. teal and orange"
                    onChange={(v) => handleCinematographyChange('look', 'color_scheme', v)}
                  />
                </div>
              )}
            </>
          )}

        </div>
        )}
      </div>
    </div>
  );
}

// ============================================================
// UI Helper Components
// ============================================================

function TabButton({ icon, active, onClick, label }: { icon: React.ReactNode; active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`p-3 rounded-xl transition-all flex flex-col items-center gap-1 w-12 ${active ? 'bg-zinc-800 text-white shadow-inner' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900'}`}
    >
      {icon}
      <span className="text-[9px] uppercase font-bold tracking-wide">{label}</span>
    </button>
  );
}

function InfoTooltip({ text }: { text: string }) {
  return (
    <div className="group relative">
      <Info size={12} className="text-zinc-600 cursor-help" />
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-black text-xs text-zinc-300 rounded border border-zinc-800 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50 shadow-xl">
        {text}
      </div>
    </div>
  );
}

function InputField({ label, value, placeholder, onChange, info }: { label: string; value: string; placeholder?: string; onChange: (val: string) => void; info?: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5">
        <label className="text-xs font-medium text-zinc-400">{label}</label>
        {info && <InfoTooltip text={info} />}
      </div>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="bg-[#0f0f11] border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500 transition-all placeholder-zinc-600"
      />
    </div>
  );
}

function TextAreaField({ label, value, placeholder, onChange, info }: { label: string; value: string; placeholder?: string; onChange: (val: string) => void; info?: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5">
        <label className="text-xs font-medium text-zinc-400">{label}</label>
        {info && <InfoTooltip text={info} />}
      </div>
      <textarea
        rows={4}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="bg-[#0f0f11] border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500 transition-all placeholder-zinc-600 resize-none"
      />
    </div>
  );
}

function SelectField({ label, value, options, onChange, info }: { label: string; value: string; options: string[]; onChange: (val: string) => void; info?: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5">
        <label className="text-xs font-medium text-zinc-400">{label}</label>
        {info && <InfoTooltip text={info} />}
      </div>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-[#0f0f11] border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-zinc-500 focus:ring-1 focus:ring-zinc-500 transition-all appearance-none cursor-pointer"
        >
          {options.map(opt => (
            <option key={opt} value={opt}>{opt.replace(/_/g, ' ')}</option>
          ))}
        </select>
        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
          <svg width="10" height="6" viewBox="0 0 10 6" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// Assembled View Component (fetches from backend)
// ============================================================

interface ReferenceImage {
  id: number;
  url: string;
  thumbnail: string | null;
  title: string | null;
}

interface AssembledViewProps {
  sceneData: SceneState;
  shortDescription: string;
  criticScore: number;
  criticIssues: string[];
  pipelineLogs: PipelineLogEntry[];
  isPipelineActive: boolean;
  showLogsView: boolean;
}

function AssembledView({ sceneData, shortDescription, criticScore, criticIssues, pipelineLogs, isPipelineActive, showLogsView }: AssembledViewProps) {
  const [assembledScene, setAssembledScene] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'json' | 'narrative'>('json');
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Track previous scene for diff highlighting
  const prevSceneRef = useRef<any>(null);
  const [changedHeads, setChangedHeads] = useState<Set<string>>(new Set());
  // Granular change tracking: which specific items changed within each head
  const [changedElements, setChangedElements] = useState<Set<string>>(new Set()); // element_ids
  const [changedObjects, setChangedObjects] = useState<Set<number>>(new Set()); // indices
  const [changedActions, setChangedActions] = useState<Set<number>>(new Set()); // indices
  const [changedCinematography, setChangedCinematography] = useState<Set<string>>(new Set()); // sub-keys: camera, lighting, composition, look
  const [prevCounts, setPrevCounts] = useState<{elements: number, objects: number, actions: number, cinematography: boolean}>({
    elements: 0, objects: 0, actions: 0, cinematography: false
  });
  const prevPipelineActiveRef = useRef<boolean>(false);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (isPipelineActive && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [pipelineLogs, isPipelineActive]);

  // Clear diff highlighting when a new pipeline starts (refinement submitted)
  useEffect(() => {
    // Detect transition from inactive to active (new pipeline starting)
    if (isPipelineActive && !prevPipelineActiveRef.current) {
      setChangedHeads(new Set());
      setChangedElements(new Set());
      setChangedObjects(new Set());
      setChangedActions(new Set());
      setChangedCinematography(new Set());
    }
    prevPipelineActiveRef.current = isPipelineActive;
  }, [isPipelineActive]);

  // Compute which heads changed when scene updates (granular tracking)
  useEffect(() => {
    if (!assembledScene) return;

    const prev = prevSceneRef.current;
    if (prev) {
      const changed = new Set<string>();
      const changedEls = new Set<string>();
      const changedObjs = new Set<number>();
      const changedActs = new Set<number>();
      const changedCin = new Set<string>();

      // Compare elements - track which specific element_ids changed
      const prevElements = prev.elements?.elements || [];
      const currElements = assembledScene.elements?.elements || [];
      const prevElMap = new Map(prevElements.map((el: any) => [el.element_id, el]));
      const currElMap = new Map(currElements.map((el: any) => [el.element_id, el]));

      // Check for changed or new elements
      currElements.forEach((el: any) => {
        const prevEl = prevElMap.get(el.element_id);
        if (!prevEl || JSON.stringify(prevEl) !== JSON.stringify(el)) {
          changedEls.add(el.element_id);
        }
      });
      // Check for removed elements
      prevElements.forEach((el: any) => {
        if (!currElMap.has(el.element_id)) {
          changedEls.add(el.element_id);
        }
      });
      if (changedEls.size > 0) changed.add('elements');

      // Compare objects - track which indices changed
      // Handle both Scene format (direct array) and SceneState format (wrapped in {objects: [...]})
      const prevObjects = Array.isArray(prev.objects) ? prev.objects : (prev.objects?.objects || []);
      const currObjects = Array.isArray(assembledScene.objects) ? assembledScene.objects : (assembledScene.objects?.objects || []);
      const maxObjLen = Math.max(prevObjects.length, currObjects.length);
      for (let i = 0; i < maxObjLen; i++) {
        if (JSON.stringify(prevObjects[i]) !== JSON.stringify(currObjects[i])) {
          changedObjs.add(i);
        }
      }
      if (changedObjs.size > 0) changed.add('objects');

      // Compare actions - track which indices changed
      // Handle both Scene format (direct array) and SceneState format (wrapped in {actions: [...]})
      const prevActions = Array.isArray(prev.actions) ? prev.actions : (prev.actions?.actions || []);
      const currActions = Array.isArray(assembledScene.actions) ? assembledScene.actions : (assembledScene.actions?.actions || []);
      const maxActLen = Math.max(prevActions.length, currActions.length);
      for (let i = 0; i < maxActLen; i++) {
        if (JSON.stringify(prevActions[i]) !== JSON.stringify(currActions[i])) {
          changedActs.add(i);
        }
      }
      if (changedActs.size > 0) changed.add('actions');

      // Compare cinematography - track which sub-components changed
      const prevCin = prev.cinematography || {};
      const currCin = assembledScene.cinematography || {};
      if (JSON.stringify(prevCin.camera) !== JSON.stringify(currCin.camera)) changedCin.add('camera');
      if (JSON.stringify(prevCin.lighting) !== JSON.stringify(currCin.lighting)) changedCin.add('lighting');
      if (JSON.stringify(prevCin.composition) !== JSON.stringify(currCin.composition)) changedCin.add('composition');
      if (JSON.stringify(prevCin.look) !== JSON.stringify(currCin.look)) changedCin.add('look');
      if (changedCin.size > 0) changed.add('cinematography');

      setChangedHeads(changed);
      setChangedElements(changedEls);
      setChangedObjects(changedObjs);
      setChangedActions(changedActs);
      setChangedCinematography(changedCin);

      // Store previous counts for Component Summary diff
      setPrevCounts({
        elements: prev.elements?.elements?.length || 0,
        objects: prev.objects?.length || 0,
        actions: prev.actions?.length || 0,
        cinematography: !!prev.cinematography
      });
    }

    prevSceneRef.current = assembledScene;
  }, [assembledScene]);

  useEffect(() => {
    // Don't fetch while pipeline is active
    if (isPipelineActive) return;

    const fetchAssembled = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/scene/assembled', {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
          },
        });
        if (!response.ok) {
          throw new Error('Failed to fetch assembled scene');
        }
        const data = await response.json();
        setAssembledScene(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setAssembledScene(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAssembled();
  }, [sceneData, isPipelineActive]);

  const displayData = assembledScene || sceneData;
  const jsonString = JSON.stringify(displayData, null, 2);

  // Render JSON with granular diff highlighting on changed items
  const renderHighlightedJson = () => {
    if (changedHeads.size === 0) {
      return <pre className="text-xs text-zinc-300 font-mono leading-relaxed whitespace-pre-wrap">{jsonString}</pre>;
    }

    const lines = jsonString.split('\n');
    let currentHead: string | null = null;
    let currentElementId: string | null = null;
    let currentArrayIndex = -1;
    let currentCinKey: string | null = null;
    let braceDepth = 0;
    let inElementsArray = false;
    let inObjectsArray = false;
    let inActionsArray = false;
    let elementStartDepth = 0;
    let objectStartDepth = 0;
    let actionStartDepth = 0;
    let cinKeyStartDepth = 0;

    return (
      <pre className="text-xs font-mono leading-relaxed whitespace-pre-wrap">
        {lines.map((line, idx) => {
          // Track brace depth for nested objects
          const openBraces = (line.match(/\{/g) || []).length;
          const closeBraces = (line.match(/\}/g) || []).length;
          const openBrackets = (line.match(/\[/g) || []).length;
          const closeBrackets = (line.match(/\]/g) || []).length;

          // Detect top-level head keys
          const headMatch = line.match(/^\s{2}"(elements|objects|actions|cinematography)":/);
          if (headMatch) {
            currentHead = headMatch[1];
            currentArrayIndex = -1;
            currentElementId = null;
            currentCinKey = null;
            if (currentHead === 'elements') inElementsArray = false;
            if (currentHead === 'objects') inObjectsArray = false;
            if (currentHead === 'actions') inActionsArray = false;
          }

          // Track entering elements.elements array
          if (currentHead === 'elements' && line.match(/^\s{4}"elements":\s*\[/)) {
            inElementsArray = true;
            currentArrayIndex = -1;
          }

          // Track entering objects array
          // Handle both Scene format ("objects": [...]) and SceneState format ("objects": { "objects": [...] })
          if (currentHead === 'objects') {
            if (line.match(/^\s{2}"objects":\s*\[/) || line.match(/^\s{4}"objects":\s*\[/)) {
              inObjectsArray = true;
              currentArrayIndex = -1;
            }
          }

          // Track entering actions array
          // Handle both Scene format ("actions": [...]) and SceneState format ("actions": { "actions": [...] })
          if (currentHead === 'actions') {
            if (line.match(/^\s{2}"actions":\s*\[/) || line.match(/^\s{4}"actions":\s*\[/)) {
              inActionsArray = true;
              currentArrayIndex = -1;
            }
          }

          // Detect element_id within elements array
          if (inElementsArray) {
            const elementIdMatch = line.match(/"element_id":\s*"([^"]+)"/);
            if (elementIdMatch) {
              currentElementId = elementIdMatch[1];
              elementStartDepth = braceDepth;
            }
          }

          // Detect new object in objects array (opening brace within objects)
          // Handle both Scene format (4-space indent) and SceneState format (6-space indent)
          if (inObjectsArray && (line.match(/^\s{4}\{/) || line.match(/^\s{6}\{/)) && !line.includes('}')) {
            currentArrayIndex++;
            objectStartDepth = braceDepth;
          }

          // Detect new action in actions array
          // Handle both Scene format (4-space indent) and SceneState format (6-space indent)
          if (inActionsArray && (line.match(/^\s{4}\{/) || line.match(/^\s{6}\{/)) && !line.includes('}')) {
            currentArrayIndex++;
            actionStartDepth = braceDepth;
          }

          // Detect cinematography sub-keys
          if (currentHead === 'cinematography') {
            const cinKeyMatch = line.match(/^\s{4}"(camera|lighting|composition|look)":/);
            if (cinKeyMatch) {
              currentCinKey = cinKeyMatch[1];
              cinKeyStartDepth = braceDepth;
            }
          }

          // Determine if this line should be highlighted
          let shouldHighlight = false;

          // Highlight changed element by element_id
          if (currentHead === 'elements' && currentElementId && changedElements.has(currentElementId)) {
            shouldHighlight = true;
          }

          // Highlight changed object by index
          if (currentHead === 'objects' && inObjectsArray && currentArrayIndex >= 0 && changedObjects.has(currentArrayIndex)) {
            shouldHighlight = true;
          }

          // Highlight changed action by index
          if (currentHead === 'actions' && inActionsArray && currentArrayIndex >= 0 && changedActions.has(currentArrayIndex)) {
            shouldHighlight = true;
          }

          // Highlight changed cinematography sub-key
          if (currentHead === 'cinematography' && currentCinKey && changedCinematography.has(currentCinKey)) {
            shouldHighlight = true;
          }

          // Also highlight the head key itself if any items in it changed
          if (headMatch && changedHeads.has(headMatch[1])) {
            const parts = line.split(`"${headMatch[1]}"`);
            braceDepth += openBraces - closeBraces + openBrackets - closeBrackets;
            return (
              <span key={idx}>
                {parts[0]}
                <span className="text-amber-400 font-semibold bg-amber-400/10 px-0.5 rounded">&quot;{headMatch[1]}&quot;</span>
                {parts[1]}
                {'\n'}
              </span>
            );
          }

          braceDepth += openBraces - closeBraces + openBrackets - closeBrackets;

          // Reset tracking when exiting arrays/objects
          if (inElementsArray && line.match(/^\s{4}\]/) && braceDepth <= 1) {
            inElementsArray = false;
            currentElementId = null;
          }
          if (inObjectsArray && line.match(/^\s{2}\]/) && braceDepth <= 0) {
            inObjectsArray = false;
            currentArrayIndex = -1;
          }
          if (inActionsArray && line.match(/^\s{2}\]/) && braceDepth <= 0) {
            inActionsArray = false;
            currentArrayIndex = -1;
          }

          // Reset element tracking when closing element object
          if (currentElementId && closeBraces > 0 && braceDepth <= elementStartDepth) {
            currentElementId = null;
          }

          // Reset cinematography sub-key tracking
          if (currentCinKey && closeBraces > 0 && braceDepth <= cinKeyStartDepth) {
            currentCinKey = null;
          }

          if (shouldHighlight) {
            return <span key={idx} className="text-amber-400 bg-amber-400/10">{line}{'\n'}</span>;
          }

          return <span key={idx} className="text-zinc-300">{line}{'\n'}</span>;
        })}
      </pre>
    );
  };

  const handleCopyContent = () => {
    const content = activeView === 'json' ? jsonString : shortDescription;
    if (content) {
      navigator.clipboard.writeText(content);
    }
  };

  const getLogColor = (type: 'info' | 'success' | 'warning' | 'error') => {
    switch (type) {
      case 'info': return 'text-cyan-400';
      case 'success': return 'text-emerald-400';
      case 'warning': return 'text-amber-400';
      case 'error': return 'text-red-400';
    }
  };

  const getLogIcon = (type: 'info' | 'success' | 'warning' | 'error') => {
    switch (type) {
      case 'info': return '→';
      case 'success': return '✓';
      case 'warning': return '⚠';
      case 'error': return '✕';
    }
  };

  // Show pipeline logs when active, when user toggled to logs view, or when there are logs but no assembled content yet
  if (isPipelineActive || showLogsView || (pipelineLogs.length > 0 && !shortDescription)) {
    return (
      <div className="animate-fadeIn flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-3 flex-shrink-0">
          <div className="flex items-center gap-2">
            <p className="text-zinc-400 text-sm">Pipeline Log</p>
            {isPipelineActive && (
              <div className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
            )}
          </div>
          <span className={`text-[10px] px-2 py-0.5 rounded-full border ${
            isPipelineActive
              ? 'bg-cyan-900/30 text-cyan-400 border-cyan-800/50'
              : 'bg-emerald-900/30 text-emerald-400 border-emerald-800/50'
          }`}>
            {isPipelineActive ? 'Running...' : 'Complete'}
          </span>
        </div>

        {/* Logs Container */}
        <div className="flex-1 min-h-0 bg-[#0a0a0c] border border-zinc-800 rounded-xl overflow-hidden">
          <div className="h-full overflow-y-auto p-4 custom-scrollbar font-mono text-xs">
            {pipelineLogs.map((log) => (
              <div key={log.id} className="mb-2 last:mb-0">
                <div className="flex items-start gap-2">
                  <span className={`shrink-0 ${getLogColor(log.type)}`}>{getLogIcon(log.type)}</span>
                  <span className={getLogColor(log.type)}>{log.message}</span>
                </div>
                {log.issues && log.issues.length > 0 && (
                  <div className="ml-5 mt-1 space-y-0.5">
                    {log.issues.map((issue, idx) => (
                      <div key={idx} className="flex items-start gap-1.5 text-amber-400/70">
                        <span className="shrink-0">•</span>
                        <span className="break-words">{issue}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>

        {/* Status footer */}
        {!isPipelineActive && (
          <div className="flex-shrink-0 pt-4">
            <p className="text-zinc-500 text-xs text-center">
              Pipeline complete. View will update to show assembled prompt.
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="animate-fadeIn flex flex-col h-full">
      {/* Scrollable Content Area */}
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-3 flex-shrink-0">
          <p className="text-zinc-400 text-sm">
            {activeView === 'json'
              ? (assembledScene ? 'Validated Scene JSON:' : 'Local scene state:')
              : 'Narrative Summary:'
            }
          </p>
          <div className="flex items-center gap-2">
            {activeView === 'narrative' && (
              <div className="relative group">
                <span className={`text-[10px] px-2 py-0.5 rounded-full border cursor-help ${
                  criticScore >= 0.85
                    ? 'bg-emerald-900/30 text-emerald-400 border-emerald-800/50'
                    : criticScore >= 0.7
                    ? 'bg-amber-900/30 text-amber-400 border-amber-800/50'
                    : 'bg-red-900/30 text-red-400 border-red-800/50'
                }`}>
                  {(criticScore * 100).toFixed(0)}% accuracy
                </span>
                {/* Hover tooltip with issues */}
                {criticIssues.length > 0 && (
                  <div className="absolute right-0 top-full mt-2 w-64 p-3 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                    <p className="text-[10px] text-amber-400 font-bold uppercase tracking-wider mb-2">
                      {criticIssues.length} Consistency {criticIssues.length === 1 ? 'Issue' : 'Issues'}
                    </p>
                    <div className="space-y-1.5 max-h-40 overflow-y-auto custom-scrollbar">
                      {criticIssues.map((issue, idx) => (
                        <div key={idx} className="flex items-start gap-1.5 text-[10px] text-zinc-400">
                          <span className="text-amber-500 mt-0.5 shrink-0">•</span>
                          <span className="break-words">{issue}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            {activeView === 'json' && assembledScene && (
              <div className="relative group">
                <span className="text-[10px] bg-emerald-900/30 text-emerald-400 px-2 py-0.5 rounded-full border border-emerald-800/50 cursor-help">
                  Backend Validated
                </span>
                {/* Hover tooltip with issues */}
                {criticIssues.length > 0 && (
                  <div className="absolute right-0 top-full mt-2 w-64 p-3 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                    <p className="text-[10px] text-amber-400 font-bold uppercase tracking-wider mb-2">
                      {criticIssues.length} Consistency {criticIssues.length === 1 ? 'Issue' : 'Issues'}
                    </p>
                    <div className="space-y-1.5 max-h-40 overflow-y-auto custom-scrollbar">
                      {criticIssues.map((issue, idx) => (
                        <div key={idx} className="flex items-start gap-1.5 text-[10px] text-zinc-400">
                          <span className="text-amber-500 mt-0.5 shrink-0">•</span>
                          <span className="break-words">{issue}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            <button
              onClick={handleCopyContent}
              className="text-[10px] px-2 py-0.5 rounded-full border bg-zinc-800 text-zinc-400 border-zinc-700 hover:text-zinc-200 hover:border-zinc-500 transition-all"
            >
              Copy
            </button>
          </div>
        </div>

        {/* Content Box - Scrollable */}
        <div className="flex-1 min-h-0 bg-[#0f0f11] border border-zinc-800 rounded-xl overflow-hidden">
          {isLoading ? (
            <div className="flex items-center justify-center h-full py-12">
              <div className="w-6 h-6 border-2 border-zinc-600 border-t-emerald-500 rounded-full animate-spin" />
            </div>
          ) : error ? (
            <div className="p-4">
              <p className="text-red-400 text-sm">{error}</p>
              <p className="text-zinc-500 text-xs mt-1">Showing local state instead.</p>
            </div>
          ) : (
            <div className="h-full overflow-y-auto p-4 custom-scrollbar">
              {activeView === 'json' ? (
                renderHighlightedJson()
              ) : activeView === 'narrative' ? (
                shortDescription ? (
                  <p className="text-zinc-200 text-sm leading-relaxed whitespace-pre-wrap">
                    {shortDescription}
                  </p>
                ) : (
                  <div className="flex items-center justify-center h-full text-zinc-500 text-sm">
                    No narrative summary available. Run Assemble first.
                  </div>
                )
              ) : null}
            </div>
          )}
        </div>
      </div>

      {/* Fixed Bottom Section */}
      <div className="flex-shrink-0 pt-4 space-y-4">
        {/* Toggle Buttons */}
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setActiveView('json')}
            className={`py-2.5 rounded-xl font-medium transition-all border text-sm ${
              activeView === 'json'
                ? 'bg-emerald-900/50 text-emerald-300 border-emerald-700 ring-1 ring-emerald-600'
                : 'bg-emerald-900/20 hover:bg-emerald-900/30 text-emerald-400 border-emerald-800/50 hover:border-emerald-700'
            }`}
          >
            JSON
          </button>
          <button
            onClick={() => setActiveView('narrative')}
            disabled={!shortDescription}
            className={`py-2.5 rounded-xl font-medium transition-all border text-sm ${
              !shortDescription
                ? 'bg-zinc-800 text-zinc-600 border-zinc-700 cursor-not-allowed'
                : activeView === 'narrative'
                ? 'bg-purple-900/50 text-purple-300 border-purple-700 ring-1 ring-purple-600'
                : 'bg-purple-900/20 hover:bg-purple-900/30 text-purple-400 border-purple-800/50 hover:border-purple-700'
            }`}
          >
            Narrative
          </button>
        </div>

        {/* Component Summary */}
        <div className="space-y-3">
          <h4 className="text-xs uppercase tracking-wider text-zinc-500 font-bold">Component Summary</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className={`bg-zinc-900/50 p-3 rounded-lg border ${changedHeads.has('elements') ? 'border-amber-500/50' : 'border-zinc-800'}`}>
              <span className="text-teal-400">Elements:</span>
              <span className={`ml-2 ${changedHeads.has('elements') ? 'text-amber-400 font-semibold' : 'text-zinc-300'}`}>
                {displayData.elements?.elements?.length || 0}
              </span>
            </div>
            <div className={`bg-zinc-900/50 p-3 rounded-lg border ${changedHeads.has('objects') ? 'border-amber-500/50' : 'border-zinc-800'}`}>
              <span className="text-blue-400">Objects:</span>
              <span className={`ml-2 ${changedHeads.has('objects') ? 'text-amber-400 font-semibold' : 'text-zinc-300'}`}>
                {displayData.objects?.length || 0}
              </span>
            </div>
            <div className={`bg-zinc-900/50 p-3 rounded-lg border ${changedHeads.has('actions') ? 'border-amber-500/50' : 'border-zinc-800'}`}>
              <span className="text-amber-400">Actions:</span>
              <span className={`ml-2 ${changedHeads.has('actions') ? 'text-orange-400 font-semibold' : 'text-zinc-300'}`}>
                {displayData.actions?.length || 0}
              </span>
            </div>
            <div className={`bg-zinc-900/50 p-3 rounded-lg border ${changedHeads.has('cinematography') ? 'border-amber-500/50' : 'border-zinc-800'}`}>
              <span className="text-purple-400">Cinematography:</span>
              <span className={`ml-2 ${changedHeads.has('cinematography') ? 'text-amber-400 font-semibold' : 'text-zinc-300'}`}>
                {displayData.cinematography ? 'Set' : 'None'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// References Sidebar Component (Freepik Image Search)
// ============================================================

interface ReferencesSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  referenceImages: Array<{id: number; url: string; thumbnail: string | null; title: string | null}>;
  onSearchImages: (query?: string) => Promise<void>;
  isSearchingImages: boolean;
  shortDescription: string;
  selectedRefId: number | null;
  onToggleSelect: (img: {id: number; url: string; thumbnail: string | null; title: string | null}) => void;
}

function ReferencesSidebar({ isOpen, onClose, referenceImages, onSearchImages, isSearchingImages, shortDescription, selectedRefId, onToggleSelect }: ReferencesSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = () => {
    const query = searchQuery.trim() || shortDescription || 'cinematic scene';
    onSearchImages(query);
  };

  return (
    <div
      className={`fixed right-0 top-0 h-full bg-zinc-950 border-l border-zinc-800 transform transition-all duration-300 ease-out z-50 ${
        isOpen ? 'translate-x-0 w-80' : 'translate-x-full w-0'
      }`}
    >
      {isOpen && (
        <div className="flex flex-col h-full p-4">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-semibold text-white">Reference Images</h2>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Search Bar */}
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder={shortDescription ? "Search or use summary..." : "Search images..."}
              className="flex-1 min-w-0 bg-zinc-900/50 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-cyan-700 focus:ring-1 focus:ring-cyan-700/50"
            />
            <button
              onClick={handleSearch}
              disabled={isSearchingImages}
              className="px-3 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors shrink-0"
            >
              {isSearchingImages ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* Results */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {isSearchingImages ? (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="w-8 h-8 border-2 border-zinc-600 border-t-cyan-500 rounded-full animate-spin mb-3" />
                <p className="text-zinc-400 text-sm">Searching reference images...</p>
              </div>
            ) : referenceImages.length > 0 ? (
              <div className="grid grid-cols-2 gap-2">
                {referenceImages.map((img, idx) => {
                  const isSelected = selectedRefId === img.id;
                  return (
                    <button
                      key={img.id}
                      onClick={() => onToggleSelect(img)}
                      className="block group text-left w-full"
                    >
                      <div className={`bg-zinc-900/50 border rounded-lg overflow-hidden transition-all relative ${
                        isSelected
                          ? 'border-cyan-500 ring-2 ring-cyan-500/30'
                          : 'border-zinc-800 hover:border-cyan-700/50'
                      }`}>
                        {img.thumbnail ? (
                          <div className="relative aspect-video">
                            <img
                              src={img.thumbnail}
                              alt={img.title || 'Reference image'}
                              className="w-full h-full object-cover"
                            />
                            <div className={`absolute inset-0 transition-colors ${
                              isSelected ? 'bg-cyan-500/20' : 'bg-cyan-500/0 group-hover:bg-cyan-500/10'
                            }`} />
                          </div>
                        ) : (
                          <div className="aspect-video bg-zinc-800 flex items-center justify-center">
                            <ImageIcon className="w-6 h-6 text-zinc-600" />
                          </div>
                        )}
                        {/* Index badge */}
                        <div className={`absolute top-1.5 left-1.5 text-[10px] font-bold px-1.5 py-0.5 rounded min-w-[20px] text-center ${
                          isSelected ? 'bg-cyan-500 text-white' : 'bg-black/70 text-white'
                        }`}>
                          {idx + 1}
                        </div>
                        {/* Selection checkmark */}
                        {isSelected && (
                          <div className="absolute top-1.5 right-1.5 bg-cyan-500 text-white rounded-full w-5 h-5 flex items-center justify-center">
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        )}
                        {img.title && (
                          <div className="p-1.5">
                            <p className={`text-[10px] line-clamp-1 transition-colors ${
                              isSelected ? 'text-cyan-400' : 'text-zinc-400 group-hover:text-cyan-400'
                            }`}>
                              {img.title}
                            </p>
                          </div>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
                <Search className="w-8 h-8 mb-3 text-zinc-600" />
                <p className="text-sm text-center px-4">
                  {shortDescription
                    ? "Enter a search term or press Search to use the scene summary"
                    : "Enter a search term to find reference images from Freepik"
                  }
                </p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="pt-4 mt-4 border-t border-zinc-800">
            <p className="text-[10px] text-zinc-500 text-center">
              Powered by Freepik API
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
