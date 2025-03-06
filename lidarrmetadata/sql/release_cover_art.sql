WITH release_info AS (
    SELECT id, gid
    FROM release
    WHERE gid = ANY($1::uuid[])
)
SELECT 
    r.gid as release_id,
    caa.id,
    caa.date_uploaded AS uploaded
FROM cover_art_archive.cover_art caa
JOIN release_info r ON r.id = caa.release
ORDER BY r.gid, caa.date_uploaded DESC;