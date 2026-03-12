import { get } from '@vercel/blob';

export async function GET({ params }: { params: { path: string } | Promise<{ path: string }> }) {
	const resolved = await params;
	const path = resolved.path;
	if (!path) {
		return new Response(JSON.stringify({ error: 'Missing path' }), {
			status: 400,
			headers: { 'Content-Type': 'application/json' },
		});
	}

	const blobPathname = `data/${path}`;

	try {
		const result = await get(blobPathname, { access: 'private' });

		if (!result) {
			return new Response(JSON.stringify({ error: 'Not found' }), {
				status: 404,
				headers: { 'Content-Type': 'application/json' },
			});
		}

		if (result.statusCode === 304) {
			return new Response(null, { status: 304 });
		}

		const contentType = result.blob.contentType ?? 'application/json';
		return new Response(result.stream, {
			status: 200,
			headers: {
				'Content-Type': contentType,
				'Cache-Control': result.blob.cacheControl ?? 'public, max-age=3600',
			},
		});
	} catch (err) {
		console.error('Blob fetch error:', err);
		return new Response(
			JSON.stringify({ error: 'Failed to fetch blob', details: String(err) }),
			{
				status: 500,
				headers: { 'Content-Type': 'application/json' },
			}
		);
	}
}
