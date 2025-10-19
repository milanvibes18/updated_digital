import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { EmailAlert } from '../types/digital-twin'
import { blink } from '../blink/client'
import { Mail, Clock, CheckCircle, XCircle, RefreshCw } from 'lucide-react'
import { Button } from './ui/button'

interface EmailLogsViewProps {
  userId: string
  onClose: () => void
}

export function EmailLogsView({ userId, onClose }: EmailLogsViewProps) {
  const [emailLogs, setEmailLogs] = useState<EmailAlert[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadEmailLogs()
    
    // Refresh every 10 seconds to show updated status
    const interval = setInterval(loadEmailLogs, 10000)
    return () => clearInterval(interval)
  }, [])

  const loadEmailLogs = async () => {
    try {
      const logs = await (blink.db as any).emailAlerts.list({
        where: { userId },
        orderBy: { sentAt: 'desc' },
        limit: 50
      })
      setEmailLogs(logs)
    } catch (error) {
      console.error('Error loading email logs:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = (status: EmailAlert['status']) => {
    switch (status) {
      case 'sent':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadgeVariant = (status: EmailAlert['status']) => {
    switch (status) {
      case 'sent':
        return 'default'
      case 'failed':
        return 'destructive'
      case 'pending':
        return 'secondary'
      default:
        return 'outline'
    }
  }

  if (loading) {
    return (
      <Card className="w-full max-w-4xl">
        <CardContent className="p-8">
          <div className="flex items-center justify-center">
            <RefreshCw className="h-8 w-8 animate-spin" />
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Email Alert Logs
            </CardTitle>
            <CardDescription>
              History of sent email notifications
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={loadEmailLogs}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {emailLogs.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Mail className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No email alerts sent yet</p>
          </div>
        ) : (
          <div className="space-y-4">
            {emailLogs.map((log) => (
              <div
                key={log.id}
                className="border rounded-lg p-4 hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="space-y-2 flex-1">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(log.status)}
                      <h4 className="font-medium">{log.subject}</h4>
                      <Badge variant={getStatusBadgeVariant(log.status)}>
                        {log.status.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div className="text-sm text-muted-foreground space-y-1">
                      <p>
                        <span className="font-medium">To:</span> {log.recipient}
                      </p>
                      <p>
                        <span className="font-medium">Sent:</span>{' '}
                        {new Date(log.sentAt).toLocaleString()}
                      </p>
                      {log.alertId && (
                        <p>
                          <span className="font-medium">Alert ID:</span> {log.alertId}
                        </p>
                      )}
                      {log.errorMessage && (
                        <p className="text-red-600">
                          <span className="font-medium">Error:</span> {log.errorMessage}
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Email body preview */}
                <details className="mt-3">
                  <summary className="cursor-pointer text-sm text-muted-foreground hover:text-foreground">
                    View email content
                  </summary>
                  <div className="mt-2 p-3 bg-muted/30 rounded text-sm">
                    <div 
                      dangerouslySetInnerHTML={{ 
                        __html: log.body.replace(/\n/g, '<br>') 
                      }} 
                    />
                  </div>
                </details>
              </div>
            ))}
            
            {emailLogs.length >= 50 && (
              <div className="text-center py-4 text-sm text-muted-foreground">
                Showing latest 50 email logs
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}