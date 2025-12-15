import { NextRequest } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const prompt = searchParams.get('prompt');

  const url = new URL(`${BACKEND_URL}/api/generate-images-stream`);
  if (prompt) url.searchParams.set('prompt', prompt);

  console.log('[Next.js API] GET /api/generate-images-stream - prompt:', prompt?.substring(0, 50));

  try {
    const response = await fetch(url.toString(), {
      headers: { 'Accept': 'text/event-stream' },
      cache: 'no-store',
    });

    if (!response.ok) {
      return new Response(JSON.stringify({ error: 'Backend error' }), {
        status: response.status,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Stream the SSE response through
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('[Next.js API] GET /api/generate-images-stream - error:', error);
    return new Response(JSON.stringify({ error: 'Backend unavailable', detail: String(error) }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
