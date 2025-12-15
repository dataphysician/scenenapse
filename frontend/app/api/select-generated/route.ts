import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    console.log('[Next.js API] POST /api/select-generated - index:', body.index);

    const response = await fetch(`${BACKEND_URL}/api/select-generated`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      cache: 'no-store',
    });

    const data = await response.json();
    console.log('[Next.js API] POST /api/select-generated - response:', response.status, 'selected:', data.selected?.index || 'none');

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Next.js API] POST /api/select-generated - error:', error);
    return NextResponse.json({ error: 'Backend unavailable', detail: String(error) }, { status: 503 });
  }
}
