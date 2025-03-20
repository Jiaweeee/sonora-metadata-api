WITH spotify_urls AS (
    SELECT id, substring(url from 33) as spotifyid
    FROM url 
    WHERE url LIKE 'https://open.spotify.com/artist/%'
)

SELECT DISTINCT ON (s.spotifyid) s.spotifyid, a.gid as mbid
FROM spotify_urls s
JOIN l_artist_url l ON l.entity1 = s.id
JOIN artist a ON l.entity0 = a.id
ORDER BY s.spotifyid, a.id  -- 对于相同的spotifyid，选择id最小的艺术家
LIMIT $1 OFFSET $2