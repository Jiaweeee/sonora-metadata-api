WITH main_artists AS (
    -- 获取主要演唱者信息
    SELECT DISTINCT
        a.gid as artist_id,
        a.name as name
    FROM track t
    JOIN recording r ON r.id = t.recording
    JOIN artist_credit ac ON ac.id = r.artist_credit
    JOIN artist_credit_name acn ON acn.artist_credit = ac.id
    JOIN artist a ON a.id = acn.artist
    WHERE t.gid = ANY($1::uuid[])
),
track_credits AS (
    -- 获取制作人员信息
    SELECT DISTINCT
        a.gid as artist_id,
        a.name as name,
        lt.name as role
    FROM track t
    JOIN recording r ON t.recording = r.id
    JOIN l_artist_recording lar ON lar.entity1 = r.id
    JOIN link l ON lar.link = l.id
    JOIN link_type lt ON l.link_type = lt.id
    JOIN artist a ON lar.entity0 = a.id
    WHERE t.gid = ANY($1::uuid[])
),
recording_urls AS (
    -- 获取录音的URL信息
    SELECT DISTINCT
        url.url,
        lt.name as type
    FROM track t
    JOIN recording r ON t.recording = r.id
    JOIN l_recording_url lru ON lru.entity0 = r.id
    JOIN link l ON l.id = lru.link
    JOIN link_type lt ON lt.id = l.link_type
    JOIN url ON url.id = lru.entity1
    WHERE t.gid = ANY($1::uuid[])
)
SELECT 
    json_build_object(
        'id', t.gid,
        'title', t.name,
        'position', t.position,
        'length', t.length,
        'artists', (
            SELECT json_agg(
                json_build_object(
                    'id', artist_id,
                    'name', name
                )
            )
            FROM main_artists
        ),
        'credits', (
            SELECT json_agg(
                json_build_object(
                    'id', artist_id,
                    'name', name,
                    'role', role
                )
            )
            FROM track_credits
        ),
        'urls', (
            SELECT json_agg(
                json_build_object(
                    'url', url,
                    'type', type
                )
            )
            FROM recording_urls
        ),
        'release', json_build_object(
            'id', r.gid,
            'title', r.name
        )
    ) as track
FROM track t
JOIN medium m ON m.id = t.medium
JOIN release r ON r.id = m.release
WHERE t.gid = ANY($1::uuid[]);