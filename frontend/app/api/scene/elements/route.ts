import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

// Disable caching
export const dynamic = 'force-dynamic';

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    console.log('[Next.js API] PUT /api/scene/elements - body:', JSON.stringify(body));

    const response = await fetch(`${BACKEND_URL}/api/scene/elements`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });

    const data = await response.json();
    console.log('[Next.js API] PUT /api/scene/elements - response:', response.status, JSON.stringify(data));

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Next.js API] PUT /api/scene/elements - error:', error);
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 503 });
  }
}
