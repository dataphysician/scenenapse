// ============================================================
// Core schema: Elements (scene ontology)
// ============================================================

export interface Element {
  element_id: string;
  role: string;
  entity_type: string;
  importance?: string;
  rough_description?: string;
}

export interface Elements {
  elements: Element[];
}

// ============================================================
// Objects head
// ============================================================

export interface Pose {
  pose_class: string;
  body_orientation: string;
  key_body_parts: string[];
  gaze_direction: string;
  expressive_face: string;
}

export interface SceneObject {
  dependencies: string[];
  category: string;
  is_primary_subject?: boolean;
  description: string;
  location: string;
  relationship?: string;
  relative_size: string;
  shape_and_color: string;
  texture: string;
  appearance_details: string;
  orientation?: string;
  pose?: Pose;
}

// ============================================================
// Actions head
// ============================================================

export interface TemporalContext {
  is_highlight_frame: boolean;
  frame_position_in_event: string;
}

export interface Action {
  dependencies: string[];
  action_class: string;
  stage_class: string;
  description: string;
  temporal_context: TemporalContext;
}

// ============================================================
// Cinematography head
// ============================================================

export interface CinematographyCamera {
  shot_size: string;
  shot_framing: string;
  camera_angle: string;
  lens_size: string;
  movement: string;
  depth_of_field: string;
  focus: string;
}

export interface CinematographyLighting {
  conditions: string;
  direction: string;
  shadows: string;
  lighting_type: string;
  mood_tag: string;
}

export interface CinematographyComposition {
  description: string;
  subject_layout: string;
}

export interface CinematographyLook {
  style_medium: string;
  artistic_style: string;
  color_scheme: string;
  mood_atmosphere: string;
  preference_score: string;
  aesthetic_score: string;
}

export interface Cinematography {
  dependencies: string[];
  camera: CinematographyCamera;
  lighting: CinematographyLighting;
  composition: CinematographyComposition;
  look: CinematographyLook;
}

// ============================================================
// Complete Scene
// ============================================================

export interface Scene {
  elements: Elements;
  objects: SceneObject[];
  actions: Action[];
  cinematography: Cinematography;
}

// ============================================================
// API Response Types
// ============================================================

export interface TimingInfo {
  stage1_elements_sec: number;
  stage2_parallel_sec: number;
  stage3_parallel_sec: number;
  total_sec: number;
}

export interface SceneResponse {
  scene: Scene;
  critic_issues: string[];
  critic_score: number;
  short_description: string;
  timing: TimingInfo;
  summary_tokens: number;
  retry_count: number;
}

export interface RefineRequest {
  scene: Scene;
  instruction: string;
  regenerate_summary?: boolean;
}

export interface RefineResponse {
  scene: Scene;
  short_description: string | null;
  affected_heads: string[];
  critic_score: number;
  critic_issues: string[];
  retry_count: number;
}

export interface AssembleResponse {
  scene: Scene;
  critic_issues: string[];
  critic_score: number;
  short_description: string;
}

// ============================================================
// App State Types
// ============================================================

export interface Attachment {
  id: string;
  type: 'text' | 'image' | 'video' | 'audio';
  name: string;
  file?: File;
  data?: string; // Base64 string without prefix
  mimeType?: string;
}

export interface SceneState {
  elements: Elements;
  objects: { objects: SceneObject[] };
  actions: { actions: Action[] };
  cinematography: Cinematography | null;
}

export type CategoryType = 'elements' | 'objects' | 'actions' | 'cinematography' | 'assembled' | null;

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  structuredData?: SceneState;
  referenceImages?: Array<{id: number; url: string; thumbnail: string | null; title: string | null}>;
}

// ============================================================
// Constants
// ============================================================

export const DEFAULT_CINEMATOGRAPHY: Cinematography = {
  dependencies: [],
  camera: {
    shot_size: "medium",
    shot_framing: "rule_of_thirds",
    camera_angle: "eye_level",
    lens_size: "35mm",
    movement: "static",
    depth_of_field: "moderate",
    focus: "sharp_on_subject"
  },
  lighting: {
    conditions: "soft natural daylight",
    direction: "frontal",
    shadows: "soft_diffused",
    lighting_type: "natural_daylight",
    mood_tag: "neutral"
  },
  composition: {
    description: "Balanced composition.",
    subject_layout: "centered"
  },
  look: {
    style_medium: "digital_cinema",
    artistic_style: "photorealistic",
    color_scheme: "natural",
    mood_atmosphere: "neutral",
    preference_score: "medium",
    aesthetic_score: "medium"
  }
};

export const SHOT_SIZES = [
  'extreme_close_up', 'close_up', 'medium_close_up', 'medium',
  'medium_long', 'long', 'extreme_long'
];

export const CAMERA_ANGLES = [
  'eye_level', 'low_angle', 'high_angle', 'dutch_angle',
  'birds_eye', 'worms_eye'
];

export const LIGHTING_CONDITIONS = [
  'soft natural daylight', 'harsh midday sun', 'golden hour warmth',
  'cool moonlight', 'neon city lights', 'candlelit interior', 'overcast diffused'
];

export const ARTISTIC_STYLES = [
  'photorealistic', 'hyperrealistic', 'impressionistic', 'noir',
  'anime', 'surrealist', 'minimalist', 'baroque', 'pop_art'
];

// ============================================================
// Image Generation Types (Fal AI)
// ============================================================

export interface GeneratedImage {
  url: string;
  width: number;
  height: number;
  content_type?: string;
  quality_score?: number | null;  // JoyQuality score (0-1)
}

export interface ImageGenerateRequest {
  prompt?: string;
  num_images?: number;
  aspect_ratio?: string;
  output_format?: string;
  resolution?: string;
}

export interface ImageGenerateResponse {
  images: GeneratedImage[];
  prompt_used: string;
  request_id: string;
}

// ============================================================
// Image Search Types (Freepik API)
// ============================================================

export interface ImageSearchResult {
  id: number;
  url: string;
  thumbnail: string | null;
  title: string | null;
}

export interface ImageSearchRequest {
  query?: string;
}

export interface ImageSearchResponse {
  images: ImageSearchResult[];
  query_used: string;
  total: number;
}
