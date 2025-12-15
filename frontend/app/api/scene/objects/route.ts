import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export const dynamic = 'force-dynamic';

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    console.log('[Next.js API] PUT /api/scene/objects - body:', JSON.stringify(body));

    const response = await fetch(`${BACKEND_URL}/api/scene/objects`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });
    const data = await response.json();
    console.log('[Next.js API] PUT /api/scene/objects - response:', response.status);

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Next.js API] PUT /api/scene/objects - error:', error);
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 503 });
  }
}
