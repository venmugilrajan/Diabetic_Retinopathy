import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    const { image, apiEndpoint } = await request.json();

    if (!image) {
      return NextResponse.json({ error: 'No image provided' }, { status: 400 });
    }

    if (!apiEndpoint) {
      return NextResponse.json({ error: 'No API endpoint specified' }, { status: 400 });
    }

    // Helper to convert base64 to Blob/Buffer for Gradio Client upload
    const dataURLtoBlob = (dataurl) => {
      const arr = dataurl.split(',');
      const mime = arr[0].match(/:(.*?);/)[1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      return new Blob([u8arr], { type: mime });
    };

    const blob = dataURLtoBlob(image);

    // Dynamically import @gradio/client (it works on Node server side too)
    const { Client } = await import('@gradio/client');

    // Connect to local port or Hugging Face Space path
    const client = await Client.connect(apiEndpoint);

    // Execute prediction
    const resData = await client.predict("/predict", [blob]);

    return NextResponse.json({ data: resData.data });
  } catch (error) {
    console.error('Server-side prediction error:', error);
    return NextResponse.json({ 
      error: error.message || 'Failed to communicate with the diagnostic backend.' 
    }, { status: 500 });
  }
}
