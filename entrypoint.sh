#!/bin/bash
set -e

# =============================================================================
# Kerberos: initial kinit + cron job to refresh every 6 hours
# SPARK_KERBEROS_KEYTAB and SPARK_KERBEROS_PRINCIPAL come from the .env file
# =============================================================================
if [ -n "$SPARK_KERBEROS_KEYTAB" ] && [ -n "$SPARK_KERBEROS_PRINCIPAL" ]; then
    echo "Running initial kinit for $SPARK_KERBEROS_PRINCIPAL ..."
    kinit -kt "$SPARK_KERBEROS_KEYTAB" "$SPARK_KERBEROS_PRINCIPAL" \
        >> /var/log/kinit.log 2>&1 || echo "Warning: initial kinit failed, check keytab"

    # Write cron job using the runtime env values (not hardcoded at build time)
    echo "0 */6 * * * root kdestroy -A; kinit -kt ${SPARK_KERBEROS_KEYTAB} ${SPARK_KERBEROS_PRINCIPAL} >> /var/log/kinit.log 2>&1" \
        > /etc/cron.d/kinit-refresh
    chmod 0644 /etc/cron.d/kinit-refresh

    service cron start
    echo "Kerberos cron refresh started."
else
    echo "Warning: SPARK_KERBEROS_KEYTAB or SPARK_KERBEROS_PRINCIPAL not set. Skipping kinit."
fi

# =============================================================================
# Start the API
# =============================================================================
exec uvicorn main:app --host 0.0.0.0 --port 3304