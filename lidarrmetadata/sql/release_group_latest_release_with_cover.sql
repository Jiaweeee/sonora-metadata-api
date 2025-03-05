SELECT r.gid AS release_id
FROM release r
JOIN release_group rg ON r.release_group = rg.id
JOIN cover_art_archive.cover_art caa ON caa.release = r.id
WHERE rg.gid = $1
AND caa.front = true
ORDER BY r.date_year DESC NULLS LAST,
         r.date_month DESC NULLS LAST,
         r.date_day DESC NULLS LAST
LIMIT 1;