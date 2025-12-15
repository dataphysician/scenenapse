import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export const dynamic = 'force-dynamic';

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    console.log('[Next.js API] PUT /api/scene/actions - body:', JSON.stringify(body));

    const response = await fetch(`${BACKEND_URL}/api/scene/actions`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });
    const data = await response.json();
    console.log('[Next.js API] PUT /api/scene/actions - response:', response.status);

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Next.js API] PUT /api/scene/actions - error:', error);
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 503 });
  }
}
