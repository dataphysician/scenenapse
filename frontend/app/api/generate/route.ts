import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    console.log('[Next.js API] POST /api/generate - formData keys:', Array.from(formData.keys()));

    const response = await fetch(`${BACKEND_URL}/api/generate`, {
      method: 'POST',
      body: formData,
      cache: 'no-store',
    });

    const data = await response.json();
    console.log('[Next.js API] POST /api/generate - response:', response.status);

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Next.js API] POST /api/generate - error:', error);
    return NextResponse.json({ error: 'Backend unavailable', detail: String(error) }, { status: 503 });
  }
}
