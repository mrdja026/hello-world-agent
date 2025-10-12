CREATE OR REPLACE FUNCTION public.fn_vendor_profiles()
RETURNS TABLE (
    vendor_id BIGINT,
    vendor_name TEXT,
    vendor_email TEXT,
    vendor_status TEXT,
    total_orders INT,
    completed_orders INT,
    pending_orders INT,
    cancelled_orders INT,
    avg_amount NUMERIC(10,2),
    last_order TIMESTAMPTZ,
    profile_summary TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.id AS vendor_id,
        v.name::text AS vendor_name,
        v.email::text AS vendor_email,
        v.status::text AS vendor_status,
        COUNT(o.id)::int AS total_orders,
        COUNT(*) FILTER (WHERE o.status = 'completed')::int AS completed_orders,
        COUNT(*) FILTER (WHERE o.status = 'pending')::int AS pending_orders,
        COUNT(*) FILTER (WHERE o.status = 'cancelled')::int AS cancelled_orders,
        COALESCE(AVG(o.amount), 0) AS avg_amount,
        MAX(o.updated_at) AS last_order,
        (
            v.name || ' is a ' || v.status || ' vendor. ' ||
            'They have handled ' || COUNT(o.id) || ' orders, ' ||
            COUNT(*) FILTER (WHERE o.status = 'completed') || ' completed, ' ||
            COUNT(*) FILTER (WHERE o.status = 'pending') || ' pending, and ' ||
            COUNT(*) FILTER (WHERE o.status = 'cancelled') || ' cancelled. ' ||
            'Average order value is ' || COALESCE(ROUND(AVG(o.amount),2),0) || '. ' ||
            'Last order was on ' || COALESCE(TO_CHAR(MAX(o.updated_at),'YYYY-MM-DD HH24:MI:SS'),'N/A')
        )::text AS profile_summary
    FROM public.vendors v
    LEFT JOIN public.orders o
        ON v.id = o.user_id
    GROUP BY v.id, v.name, v.email, v.status
    ORDER BY v.id;
END;
$$;
