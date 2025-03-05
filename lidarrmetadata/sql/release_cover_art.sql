WITH release_info AS (
    SELECT id
    FROM release
    WHERE gid = $1::uuid
)
SELECT 
    caa.id,
    caa.date_uploaded AS uploaded
FROM cover_art_archive.cover_art caa
JOIN release_info r ON r.id = caa.release
ORDER BY caa.date_uploaded DESC
LIMIT 1;