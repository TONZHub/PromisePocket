package com.mosslet.promisepocket.worker

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.mosslet.promisepocket.MainActivity
import com.mosslet.promisepocket.data.local.AppDatabase
import com.mosslet.promisepocket.data.model.CommitmentStatus
import com.mosslet.promisepocket.data.remote.CommitmentRemoteMapper
import com.mosslet.promisepocket.data.remote.MobileSessionStore
import com.mosslet.promisepocket.data.remote.ReceiptsMobileClient

class ReceiptsSyncWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        val sessionStore = MobileSessionStore(applicationContext)
        if (!sessionStore.isLinked) {
            return Result.success()
        }

        val mobileClient = ReceiptsMobileClient(applicationContext)
        val database = AppDatabase.getDatabase(applicationContext)
        val dao = database.commitmentDao()

        return try {
            val response = mobileClient.service.listCommitments()
            response.items?.forEach { remote ->
                val entity = CommitmentRemoteMapper.toEntity(remote)
                val local = dao.getById(entity.actorId, entity.commitmentId)
                if (local == null) {
                    dao.insert(entity)
                } else {
                    dao.update(entity)
                }
            }

            // Evaluate attention items
            val actorId = sessionStore.actorId ?: return Result.success()
            val allItems = dao.getListForActor(actorId)
            val candidateCount = allItems.count { it.status == CommitmentStatus.CANDIDATE }
            val overdueCount = allItems.count { it.status == CommitmentStatus.OVERDUE }
            val likelyDoneCount = allItems.count { it.status == CommitmentStatus.LIKELY_DONE }
            val totalAttention = candidateCount + overdueCount + likelyDoneCount

            if (totalAttention > 0) {
                val parts = mutableListOf<String>()
                if (candidateCount > 0) {
                    parts.add(if (candidateCount == 1) "1 candidate" else "$candidateCount candidates")
                }
                if (overdueCount > 0) {
                    parts.add("$overdueCount overdue")
                }
                if (likelyDoneCount > 0) {
                    parts.add(if (likelyDoneCount == 1) "1 evidence check" else "$likelyDoneCount evidence checks")
                }
                val summary = parts.joinToString(", ") + " require your confirmation."
                showAttentionNotification(applicationContext, summary)
            }

            Result.success()
        } catch (e: Exception) {
            e.printStackTrace()
            Result.retry()
        }
    }

    companion object {
        // New channel ID intentionally forces Android 8+ to create the channel
        // at urgent importance. Existing channel importance cannot be raised
        // programmatically after the channel has already been created.
        const val CHANNEL_ID = "receipts_attention_urgent_v2"
        const val NOTIFICATION_ID = 1001

        fun createNotificationChannel(context: Context) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val name = "Receipts Urgent Alerts"
                val descriptionText = "Urgent alerts when candidate promises need confirmation or promises are overdue"
                val channel = NotificationChannel(
                    CHANNEL_ID,
                    name,
                    NotificationManager.IMPORTANCE_HIGH
                ).apply {
                    description = descriptionText
                    enableVibration(true)
                    setShowBadge(true)
                }
                val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.createNotificationChannel(channel)
            }
        }

        fun showAttentionNotification(context: Context, bodyText: String) {
            createNotificationChannel(context)

            val intent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            }
            val pendingIntent = PendingIntent.getActivity(
                context,
                0,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val builder = NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle("Receipts: Attention Required")
                .setContentText(bodyText)
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setCategory(NotificationCompat.CATEGORY_REMINDER)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)

            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(NOTIFICATION_ID, builder.build())
        }
    }
}
