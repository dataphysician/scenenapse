import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    // Get the FormData from the request
    const formData = await request.formData();
    const instruction = formData.get('instruction');
    const hasImage = formData.has('image');

    console.log('[Next.js API] POST /api/refine - instruction:', instruction, 'hasImage:', hasImage);

    // Forward FormData to backend
    const response = await fetch(`${BACKEND_URL}/api/refine`, {
      method: 'POST',
      body: formData,
      cache: 'no-store',
    });

    const data = await response.json();
    console.log('[Next.js API] POST /api/refine - response:', response.status);

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Next.js API] POST /api/refine - error:', error);
    return NextResponse.json({ error: 'Backend unavailable', detail: String(error) }, { status: 503 });
  }
}
