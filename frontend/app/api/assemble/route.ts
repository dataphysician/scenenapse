import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    console.log('[Next.js API] POST /api/assemble - validating existing scene');

    const response = await fetch(`${BACKEND_URL}/api/assemble`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });

    const data = await response.json();
    console.log('[Next.js API] POST /api/assemble - response:', response.status);

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Next.js API] POST /api/assemble - error:', error);
    return NextResponse.json({ error: 'Backend unavailable', detail: String(error) }, { status: 503 });
  }
}
